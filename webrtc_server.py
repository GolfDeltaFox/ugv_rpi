import asyncio
import logging
import json
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
from av import VideoFrame
import numpy as np
import time
from picamera2 import Picamera2

# Configure clean logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("webrtc")

class CameraStream(VideoStreamTrack):
    def __init__(self, picam2, cv_ctrl=None):
        super().__init__()
        self.picam2 = picam2
        self.cv_ctrl = cv_ctrl
        self.last_frame_time = time.time()

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for recording if cv_ctrl is available
        if self.cv_ctrl:
            self.cv_ctrl.process_frame(frame)
            
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class WebRTCServer:
    def __init__(self, cv_ctrl_instance=None):
        self.pcs = set()
        self.shared_picam2 = None
        self.camera_lock = asyncio.Lock()
        self.app = None
        self.runner = None
        self.cv_ctrl = cv_ctrl_instance  # Store reference to cv_ctrl instance

    async def offer(self, request):
        logger.info("[offer] Received offer from client.")
        try:
            params = await request.json()
            if not params.get("sdp") or not params.get("type"):
                raise ValueError("Missing 'sdp' or 'type'")
        except Exception as e:
            logger.error(f"[offer] Invalid payload: {e}")
            return web.json_response({"error": str(e)}, status=400)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        # Reuse the camera instance from cv_ctrl if available
        async with self.camera_lock:
            if self.shared_picam2 is None:
                if self.cv_ctrl and self.cv_ctrl.picam2:
                    self.shared_picam2 = self.cv_ctrl.picam2
                    logger.info("Using existing camera from cv_ctrl")
                else:
                    self.shared_picam2 = Picamera2()
                    self.shared_picam2.configure(self.shared_picam2.create_video_configuration(
                        main={"format": 'XRGB8888', "size": (640, 480)}
                    ))
                    self.shared_picam2.start()
                    logger.info("Camera started")

        # Create a new track for this connection
        pc.addTrack(CameraStream(self.shared_picam2, self.cv_ctrl))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("[offer] WebRTC negotiation successful.")
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }, headers={
            "Access-Control-Allow-Origin": "*"
        })

    async def options_handler(self, request):
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    async def on_shutdown(self, app):
        logger.info("Shutting down...")
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    async def _run_webrtc_server(self):
        self.app = web.Application()
        self.app.router.add_post("/offer", self.offer)
        self.app.router.add_options("/offer", self.options_handler)
        self.app.on_shutdown.append(self.on_shutdown)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, port=8081)
        await site.start()
        logger.info("WebRTC server running on port 8081")

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_webrtc_server())
        loop.run_forever()

# Update the backward compatibility function
def start_webrtc_server(cv_ctrl_instance=None):
    server = WebRTCServer(cv_ctrl_instance)
    server.start()
