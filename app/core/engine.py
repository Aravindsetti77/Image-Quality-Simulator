import cv2
import numpy as np
import onnxruntime as ort
import os

class PirateEngine:
    def __init__(self, model_path="realesr-general-x4v3.onnx"):
        self.model_path = model_path
        self.ort_session = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please run download_model.py")
        if self.ort_session is None:
            # CPU execution provider to keep RAM low, or CUDA if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)

    def _add_noise(self, image, percentage):
        row, col, ch = image.shape
        mean = 0
        var = 255 * percentage
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def process_camrip(self, image):
        h, w = image.shape[:2]
        # 60% downscale
        small = cv2.resize(image, (int(w * 0.4), int(h * 0.4)), interpolation=cv2.INTER_AREA)
        # 11x11 Gaussian blur
        blurred = cv2.GaussianBlur(small, (11, 11), 0)
        # Contrast flatting (alpha 0.6, beta 40)
        contrast = cv2.convertScaleAbs(blurred, alpha=0.6, beta=40)
        # Add 5% noise
        noisy = self._add_noise(contrast, 0.05)
        # Resize back to original
        return cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_telesync(self, image):
        h, w = image.shape[:2]
        # 80% downscale
        small = cv2.resize(image, (int(w * 0.2), int(h * 0.2)), interpolation=cv2.INTER_AREA)
        # 5x5 blur
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        # Contrast (alpha 0.8, beta 20)
        contrast = cv2.convertScaleAbs(blurred, alpha=0.8, beta=20)
        # 2% noise
        noisy = self._add_noise(contrast, 0.02)
        return cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_telecine(self, image):
        # Color space manipulation (yellow/green tint)
        b, g, r = cv2.split(image)
        g = cv2.add(g, 20)
        r = cv2.add(r, 10)
        b = cv2.subtract(b, 20)
        tinted = cv2.merge((b, g, r))
        
        # Add random vertical scratch lines
        h, w = tinted.shape[:2]
        for _ in range(np.random.randint(2, 5)):
            x = np.random.randint(0, w)
            cv2.line(tinted, (x, 0), (x, h), (0, 0, 0), 1)
        
        return tinted

    def process_screener(self, image):
        h, w = image.shape[:2]
        text = "FOR PROMOTIONAL USE ONLY"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 800.0)
        thickness = max(1, int(scale * 2))
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        # Overlay semi-transparent text
        overlay = image.copy()
        cv2.putText(overlay, text, ((w - text_w) // 2, h - 50), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Mild JPEG compression (Quality 40)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_dvdrip(self, image):
        h, w = image.shape[:2]
        # Hard resize to 640x360
        resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        # JPEG encode quality 8
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 8]
        _, encimg = cv2.imencode('.jpg', resized, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        # Resize back using INTER_NEAREST
        return cv2.resize(decimg, (w, h), interpolation=cv2.INTER_NEAREST)

    def process_hdtv(self, image):
        # Interlacing artifact
        h, w = image.shape[:2]
        interlaced = image.copy()
        interlaced[1::2, :] = interlaced[1::2, :] * 0.8
        interlaced = interlaced.astype(np.uint8)

        # Fake channel logo watermark
        logo_size = max(20, w // 20)
        cv2.rectangle(interlaced, (w - logo_size - 20, 20), (w - 20, logo_size + 20), (255, 255, 255), -1)
        cv2.putText(interlaced, "TV", (w - logo_size - 15, logo_size + 5), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, logo_size/40.0), (0, 0, 0), 2)
        return interlaced

    def process_webrip(self, image):
        # Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        # Posterization (truncate least significant bits)
        posterized = (filtered // 32) * 32
        return posterized.astype(np.uint8)

    def process_yify(self, image):
        # Extreme bilateral filter
        filtered = cv2.bilateralFilter(image, 15, 100, 100)
        # Heavy macroblocking (Quality 15)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
        _, encimg = cv2.imencode('.jpg', filtered, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_bdrip(self, image):
        h, w = image.shape[:2]
        # Gentle downscale to 720p if larger
        if h > 720:
            new_w = int(w * (720 / h))
            image = cv2.resize(image, (new_w, 720), interpolation=cv2.INTER_AREA)
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        res = cv2.imdecode(encimg, 1)
        
        # resize back for consistency if needed
        if h > 720:
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
        return res

    def process_remux(self, image):
        self.load_model()
        
        # ONNX model input requires [1, C, H, W] float32 in [0, 1]
        input_image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        input_image = np.transpose(input_image, (2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        # Inference
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        result = self.ort_session.run([output_name], {input_name: input_image})[0]
        
        # Process output
        output_image = np.squeeze(result, axis=0)
        output_image = np.clip(output_image, 0.0, 1.0)
        output_image = np.transpose(output_image, (1, 2, 0))
        output_image = (output_image * 255.0).astype(np.uint8)
        
        return output_image

    def process(self, image, tier, resolution="none", hdr=False):
        tier = tier.lower()
        if tier == "none":
            res = image.copy()
        elif tier == "camrip":
            res = self.process_camrip(image)
        elif tier == "telesync" or tier == "ts":
            res = self.process_telesync(image)
        elif tier == "telecine" or tier == "tc":
            res = self.process_telecine(image)
        elif tier == "screener" or tier == "scr":
            res = self.process_screener(image)
        elif tier == "dvdrip":
            res = self.process_dvdrip(image)
        elif tier == "hdtv":
            res = self.process_hdtv(image)
        elif tier == "webrip":
            res = self.process_webrip(image)
        elif tier == "yify":
            res = self.process_yify(image)
        elif tier == "bdrip":
            res = self.process_bdrip(image)
        elif tier == "remux":
            res = self.process_remux(image)
        else:
            raise ValueError(f"Unknown format tier: {tier}")
            
        if resolution != "none" and resolution != "original":
            target_h = int(resolution)
            h, w = res.shape[:2]
            if target_h != h:
                target_w = int(w * (target_h / h))
                res = cv2.resize(res, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
        if hdr:
            res = cv2.detailEnhance(res, sigma_s=12, sigma_r=0.15)
                
        return res
