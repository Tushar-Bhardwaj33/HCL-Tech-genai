from unstructured.documents.elements import Table, CompositeElement, Image

__all__ = ["GET"]

class GET:
    @staticmethod
    def get_available_methods():
        return [
            GET.get_images,
            GET.get_tables,
            GET.get_texts,
            GET.get_metadata,
        ]
    
    @staticmethod
    def get_tables(chunks):
        tables = [chunk for chunk in chunks if isinstance(chunk, Table)]
        return tables
    
    @staticmethod
    def get_texts(chunks):
        texts = [chunk for chunk in chunks if isinstance(chunk, CompositeElement)]
        return texts

    @staticmethod
    def get_images(chunks):
        images_b64 = []
        for chunk in chunks:
            if isinstance(chunk, CompositeElement):
                orig_elements = getattr(chunk.metadata, "orig_elements", [])
                for el in orig_elements:
                    if isinstance(el, Image):
                        images_b64.append(el.metadata.image_base64)

        return images_b64