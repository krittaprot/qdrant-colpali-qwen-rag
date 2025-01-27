import fitz
from typing import List, Optional
from PIL import Image
import os

class PDFProcessor:
    def __init__(self, storage_dir: str = "stored_images"):
        """
        Initialize PDFProcessor with storage directory
        
        Args:
            storage_dir (str): Directory to store extracted images
        """
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
        
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
    def _generate_filename(self, pdf_base_name: str, page_num: int) -> str:
        """
        Generate unique filename for stored image
        
        Args:
            pdf_base_name (str): Base name of the PDF file
            page_num (int): Page number
            
        Returns:
            str: Unique filename
        """
        return f"{pdf_base_name}_page_{page_num}.png"
    
    def _get_existing_filename(self, pdf_base_name: str, page_num: int) -> Optional[str]:
        """
        Check if an image for the given PDF and page number already exists.
        
        Args:
            pdf_base_name (str): Base name of the PDF file
            page_num (int): Page number
            
        Returns:
            Optional[str]: Filename if it exists, None otherwise
        """
        for filename in os.listdir(self.storage_dir):
            if f"_{page_num}.png" in filename and pdf_base_name in filename:
                return filename
        return None
    
    def convert_and_store_pdf(self, 
                              pdf_file, 
                              zoom_x: float = 2, 
                              zoom_y: float = 2) -> List[dict]:
        """
        Convert PDF to images and store them locally
        
        Args:
            pdf_file: PDF file object
            zoom_x (float): Horizontal zoom factor
            zoom_y (float): Vertical zoom factor
            
        Returns:
            List[dict]: List of dictionaries containing image metadata
        """
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        stored_images = []
        metadata = []
        
        pdf_name = getattr(pdf_file, 'name', 'unnamed_pdf')
        pdf_base_name = os.path.splitext(os.path.basename(pdf_name))[0]
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Generate the filename and save image
            filename = self._generate_filename(pdf_base_name, page_num)
            filepath = os.path.join(self.storage_dir, filename)
            
            identical_flag = False
            
            # Check if the image already exists before saving
            existing_filename = self._get_existing_filename(pdf_base_name, page_num)
            if existing_filename:
                identical_flag = True
            else:
                image.save(filepath, "PNG")
            
            stored_images.append(image)

            # Store metadata
            metadata.append({
                'page_num': page_num,
                'filename': filename,
                'filepath': filepath,
                'width': pix.width,
                'height': pix.height,
            })

        return stored_images, metadata, identical_flag
        
    def get_image_path(self, filename: str) -> Optional[str]:
        """
        Get full path of stored image
        
        Args:
            filename (str): Name of the image file
            
        Returns:
            Optional[str]: Full path to image if exists, None otherwise
        """
        filepath = os.path.join(self.storage_dir, filename)
        return filepath if os.path.exists(filepath) else None
    
    def display_image(self, filename: str) -> Optional[Image.Image]:
        """
        Load and return stored image
        
        Args:
            filename (str): Name of the image file
            
        Returns:
            Optional[Image.Image]: PIL Image object if exists, None otherwise
        """
        filepath = self.get_image_path(filename)
        if filepath:
            return Image.open(filepath)
        return None