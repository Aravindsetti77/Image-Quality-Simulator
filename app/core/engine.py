import cv2
import numpy as np
import onnxruntime as ort
import os
import gc

class QualityEngine:
    def __init__(self, model_path="EDSR_x4.pb"):
        self.model_path = model_path
        self.sr = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}.")
        if self.sr is None:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(self.model_path)
            self.sr.setModel('edsr', 4)

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
        small = cv2.resize(image, (int(w * 0.4), int(h * 0.4)), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (11, 11), 0)
        contrast = cv2.convertScaleAbs(blurred, alpha=0.6, beta=40)
        noisy = self._add_noise(contrast, 0.05)
        return cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_hdcam(self, image):
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * 0.6), int(h * 0.6)), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (7, 7), 0)
        contrast = cv2.convertScaleAbs(blurred, alpha=0.7, beta=30)
        noisy = self._add_noise(contrast, 0.03)
        return cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_hdts(self, image):
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        contrast = cv2.convertScaleAbs(blurred, alpha=0.8, beta=15)
        noisy = self._add_noise(contrast, 0.02)
        return cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_telesync(self, image):
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * 0.2), int(h * 0.2)), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        contrast = cv2.convertScaleAbs(blurred, alpha=0.8, beta=20)
        noisy = self._add_noise(contrast, 0.02)
        return cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_workprint(self, image):
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = (hsv[..., 1] * 0.7).astype(np.uint8)
        desaturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, w / 1000.0)
        cv2.putText(desaturated, "01:04:32:15", (w // 2 - 50, h - 30), font, scale, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(desaturated, "PROPERTY OF STUDIO", (w // 2 - 100, 30), font, scale, (200, 200, 200), 1, cv2.LINE_AA)
        return desaturated

    def process_vhsrip(self, image):
        b, g, r = cv2.split(image)
        h, w = image.shape[:2]
        r_shifted = np.roll(r, 5, axis=1)
        b_shifted = np.roll(b, -5, axis=1)
        merged = cv2.merge((b_shifted, g, r_shifted))
        
        for i in range(0, h, max(1, h // 10)):
            if np.random.rand() > 0.7:
                merged[i:i+5, :] = np.clip(merged[i:i+5, :].astype(np.uint16) * 0.5 + 100, 0, 255).astype(np.uint8)
        
        return self._add_noise(merged, 0.08)

    def process_telecine(self, image):
        b, g, r = cv2.split(image)
        g = cv2.add(g, 20)
        r = cv2.add(r, 10)
        b = cv2.subtract(b, 20)
        tinted = cv2.merge((b, g, r))
        
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
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        overlay = image.copy()
        cv2.putText(overlay, text, ((w - text_w) // 2, h - 50), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_r5(self, image):
        b, g, r = cv2.split(image)
        g = cv2.add(g, 10)
        tinted = cv2.merge((b, g, r))
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
        _, encimg = cv2.imencode('.jpg', tinted, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_dvdrip(self, image):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 8]
        _, encimg = cv2.imencode('.jpg', resized, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return cv2.resize(decimg, (w, h), interpolation=cv2.INTER_NEAREST)

    def process_pdtv(self, image):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (720, 480), interpolation=cv2.INTER_AREA)
        interlaced = resized.copy()
        interlaced[1::2, :] = interlaced[1::2, :] * 0.7
        interlaced = interlaced.astype(np.uint8)
        
        logo_size = max(15, w // 25)
        cv2.rectangle(interlaced, (720 - logo_size - 10, 10), (720 - 10, logo_size + 10), (255, 255, 255), -1)
        cv2.putText(interlaced, "PDTV", (720 - logo_size - 5, logo_size), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return cv2.resize(interlaced, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_hdtv(self, image):
        h, w = image.shape[:2]
        interlaced = image.copy()
        interlaced[1::2, :] = interlaced[1::2, :] * 0.8
        interlaced = interlaced.astype(np.uint8)

        logo_size = max(20, w // 20)
        cv2.rectangle(interlaced, (w - logo_size - 20, 20), (w - 20, logo_size + 20), (255, 255, 255), -1)
        cv2.putText(interlaced, "TV", (w - logo_size - 15, logo_size + 5), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, logo_size/40.0), (0, 0, 0), 2)
        return interlaced

    def process_webrip(self, image):
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        posterized = (filtered // 32) * 32
        return posterized.astype(np.uint8)

    def process_hdrip(self, image):
        filtered = cv2.bilateralFilter(image, 7, 50, 50)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        _, encimg = cv2.imencode('.jpg', filtered, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_web_dl(self, image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_yify(self, image):
        filtered = cv2.bilateralFilter(image, 15, 100, 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
        _, encimg = cv2.imencode('.jpg', filtered, encode_param)
        return cv2.imdecode(encimg, 1)

    def process_bdscr(self, image):
        h, w = image.shape[:2]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        res = cv2.imdecode(encimg, 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 700.0)
        thickness = max(1, int(scale * 1.5))
        cv2.putText(res, "STUDIO PROPERTY - DO NOT DISTRIBUTE", (10, h - 30), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
        
        if np.random.rand() > 0.9:
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        return res

    def process_brrip(self, image):
        h, w = image.shape[:2]
        if h > 1080:
            new_w = int(w * (1080 / h))
            image = cv2.resize(image, (new_w, 1080), interpolation=cv2.INTER_AREA)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        res = cv2.imdecode(encimg, 1)
        
        if h > 1080:
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
        return res

    def process_bdrip(self, image):
        h, w = image.shape[:2]
        if h > 720:
            new_w = int(w * (720 / h))
            image = cv2.resize(image, (new_w, 720), interpolation=cv2.INTER_AREA)
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        res = cv2.imdecode(encimg, 1)
        
        if h > 720:
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
        return res

    def process_remux(self, image, resolution="none"):
        try:
            self.load_model()
            output_image = self.sr.upsample(image)
            return output_image
        except Exception:
            # Fallback upscaling method if the EDSR model is missing or runs out of memory
            h, w = image.shape[:2]
            if resolution != "none" and resolution != "original":
                target_h = int(resolution)
                if target_h > h:
                    target_w = int(w * (target_h / h))
                    upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    upscaled = image.copy()
            else:
                upscaled = cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
            
            try:
                return cv2.detailEnhance(upscaled, sigma_s=10, sigma_r=0.15)
            except Exception:
                return upscaled

    def process_uhd_remux(self, image):
        h, w = image.shape[:2]
        target_h = 2160
        if h != target_h:
            target_w = int(w * (target_h / h))
            upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            upscaled = image.copy()
        
        res = cv2.detailEnhance(upscaled, sigma_s=12, sigma_r=0.15)
        return res

    def process(self, image, tier, resolution="none", hdr=False):
        tier = tier.lower()
        if tier == "none":
            res = image.copy()
        elif tier == "camrip":
            res = self.process_camrip(image)
        elif tier == "hdcam":
            res = self.process_hdcam(image)
        elif tier == "telesync" or tier == "ts":
            res = self.process_telesync(image)
        elif tier == "hdts":
            res = self.process_hdts(image)
        elif tier == "workprint" or tier == "wp":
            res = self.process_workprint(image)
        elif tier == "vhsrip":
            res = self.process_vhsrip(image)
        elif tier == "telecine" or tier == "tc":
            res = self.process_telecine(image)
        elif tier == "screener" or tier == "scr":
            res = self.process_screener(image)
        elif tier == "r5":
            res = self.process_r5(image)
        elif tier == "dvdrip":
            res = self.process_dvdrip(image)
        elif tier == "pdtv":
            res = self.process_pdtv(image)
        elif tier == "hdtv":
            res = self.process_hdtv(image)
        elif tier == "webrip":
            res = self.process_webrip(image)
        elif tier == "hdrip":
            res = self.process_hdrip(image)
        elif tier == "web-dl":
            res = self.process_web_dl(image)
        elif tier == "yify":
            res = self.process_yify(image)
        elif tier == "bdscr":
            res = self.process_bdscr(image)
        elif tier == "brrip":
            res = self.process_brrip(image)
        elif tier == "bdrip":
            res = self.process_bdrip(image)
        elif tier == "remux":
            res = self.process_remux(image, resolution)
        elif tier == "uhd-remux":
            res = self.process_uhd_remux(image)
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
