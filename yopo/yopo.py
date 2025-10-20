from pathlib import Path
from yopo.models import modal_sizes 
from typing import Optional, Union
from ultralytics import YOLO
import requests
from PIL import Image
import numpy.typing as npt

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
            source: Union[str, int, Image, npt.NDArray]
        ):
        if(not self.modal):
            raise RuntimeError('no modal is initialized')
        return self.modal.predict(
            source=source
        )
