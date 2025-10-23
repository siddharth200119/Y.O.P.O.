from pathlib import Path
from yopo.models import modal_sizes 
from typing import List, Optional
from ultralytics import YOLO
import requests
from yopo.models import Result, BBOX
from PIL import Image
import pytesseract

class YOPO():
    def __init__(
        self,
        modal_size: Optional[modal_sizes] = None,
        version: Optional[str] = None,
        modal_path: Optional[Path] = None
    ):

        if(not modal_size and not modal_path):
            raise ValueError('Please provide a modal size like n, s, m, etc or a modal path')
        
        self.version = version
        self.modal_size = modal_size
        self.modal_path = modal_path
        self.modal = YOLO(modal_path or self.load_modal())
    
    def get_cache_dir(self) -> Path:
        cache_dir = Path.home() / ".cache" / "y_o_p_o" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def load_modal(self) -> Path:
        if not self.version:
            #logic to get the latest version using github api
            api_response = requests.get('https://api.github.com/repos/siddharth200119/Y.O.P.O./releases/latest')
            api_response.raise_for_status()
            response_data = api_response.json()
            self.version = response_data.get('tag_name', None)
            if(not self.version):
                raise RuntimeError('No latest versions found please contact developer or pass in a custom modal')

        link = f"https://github.com/siddharth200119/Y.O.P.O./releases/download/{self.version}/yopo_{self.modal_size}.pt"
        file_name = f"yopo_{self.version}_{self.modal_size}.pt"

        file_path : Path = self.get_cache_dir() / file_name

        if(file_path.exists()):
            print(f'loading modal form cache: {file_path}')
            return file_path

        print(f'Downloading modal and saving to cache')
        response = requests.get(link)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8091):
                f.write(chunk)

        return file_path

    def predict(
            self,
            source: Path,
            ocr_level: str = 'block'  # 'word', 'line', 'paragraph', or 'block'
        ):
        if(not self.modal):
            raise RuntimeError('no modal is initialized')
        yolo_results = self.modal.predict(
            source=source
        )[0]
        image = Image.open(source)
        ocr_results = pytesseract.image_to_data(
            image, 
            output_type=pytesseract.Output.DICT,
            config='--psm 6'
        )
        
        final_results: List[Result] = []
        
        # Process YOLO results
        for box in yolo_results.boxes:
            bbox = BBOX(
                xyxy=box.xyxy[0].tolist(),
                xywh=box.xywh[0].tolist()
            )
            final_results.append(
                Result(
                    bbox=bbox,
                    class_idx=int(box.cls[0].item()),
                    conf=float(box.conf[0].item()),
                    content=None
                )
            )
        
        # Group OCR results based on level
        if ocr_level == 'word':
            # Original word-level processing
            n_boxes = len(ocr_results['text'])
            for i in range(n_boxes):
                if int(ocr_results['conf'][i]) > 0 and ocr_results['text'][i].strip():
                    x, y, w, h = (
                        ocr_results['left'][i],
                        ocr_results['top'][i],
                        ocr_results['width'][i],
                        ocr_results['height'][i]
                    )
                    
                    xyxy = [x, y, x + w, y + h]
                    xywh = [x + w/2, y + h/2, w, h]
                    
                    bbox = BBOX(xyxy=xyxy, xywh=xywh)
                    
                    final_results.append(
                        Result(
                            bbox=bbox,
                            class_idx=-1,
                            conf=float(ocr_results['conf'][i]) / 100.0,
                            content=ocr_results['text'][i]
                        )
                    )
        else:
            # Group by line, paragraph, or block
            grouped_text = {}
            n_boxes = len(ocr_results['text'])
            
            for i in range(n_boxes):
                if int(ocr_results['conf'][i]) > 0 and ocr_results['text'][i].strip():
                    # Determine grouping key based on level
                    if ocr_level == 'line':
                        group_key = (
                            ocr_results['block_num'][i],
                            ocr_results['par_num'][i],
                            ocr_results['line_num'][i]
                        )
                    elif ocr_level == 'paragraph':
                        group_key = (
                            ocr_results['block_num'][i],
                            ocr_results['par_num'][i]
                        )
                    else:  # block
                        group_key = (ocr_results['block_num'][i],)
                    
                    if group_key not in grouped_text:
                        grouped_text[group_key] = {
                            'text': [],
                            'left': [],
                            'top': [],
                            'right': [],
                            'bottom': [],
                            'conf': []
                        }
                    
                    x, y, w, h = (
                        ocr_results['left'][i],
                        ocr_results['top'][i],
                        ocr_results['width'][i],
                        ocr_results['height'][i]
                    )
                    
                    grouped_text[group_key]['text'].append(ocr_results['text'][i])
                    grouped_text[group_key]['left'].append(x)
                    grouped_text[group_key]['top'].append(y)
                    grouped_text[group_key]['right'].append(x + w)
                    grouped_text[group_key]['bottom'].append(y + h)
                    grouped_text[group_key]['conf'].append(int(ocr_results['conf'][i]))
            
            # Create bounding boxes for grouped text
            for group_data in grouped_text.values():
                if not group_data['text']:
                    continue
                
                combined_text = ' '.join(group_data['text'])
                
                x1 = min(group_data['left'])
                y1 = min(group_data['top'])
                x2 = max(group_data['right'])
                y2 = max(group_data['bottom'])
                
                w = x2 - x1
                h = y2 - y1
                
                avg_conf = sum(group_data['conf']) / len(group_data['conf'])
                
                xyxy = [x1, y1, x2, y2]
                xywh = [x1 + w/2, y1 + h/2, w, h]
                
                bbox = BBOX(xyxy=xyxy, xywh=xywh)
                
                final_results.append(
                    Result(
                        bbox=bbox,
                        class_idx=-1,
                        conf=float(avg_conf) / 100.0,
                        content=combined_text
                    )
                )
        
        return final_results
