from .core import OCRBox, PPTProject, PPTSlide, convert_pdf_to_pptx, export_editable_ppt, prepare_project, save_project_cache
from .version import APP_EXE_NAME, APP_NAME, APP_PUBLISHER, APP_VERSION

__all__ = [
    "OCRBox",
    "PPTProject",
    "PPTSlide",
    "APP_EXE_NAME",
    "APP_NAME",
    "APP_PUBLISHER",
    "APP_VERSION",
    "convert_pdf_to_pptx",
    "export_editable_ppt",
    "prepare_project",
    "save_project_cache",
]
