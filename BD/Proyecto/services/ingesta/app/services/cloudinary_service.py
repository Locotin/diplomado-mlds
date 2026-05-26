from typing import Any

import anyio
import cloudinary
import cloudinary.uploader

from app.models import CloudinaryAsset


class CloudinaryService:
    def __init__(self, *, cloud_name: str, api_key: str, api_secret: str, folder: str) -> None:
        self._folder = folder
        cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True)

    async def upload_image(self, *, file_bytes: bytes, filename: str, source_id: str) -> CloudinaryAsset:
        upload_result = await anyio.to_thread.run_sync(
            lambda: cloudinary.uploader.upload(
                file_bytes,
                resource_type="image",
                folder=self._folder,
                public_id=source_id,
                overwrite=True,
            )
        )
        return CloudinaryAsset(
            asset_id=upload_result["asset_id"],
            public_id=upload_result["public_id"],
            secure_url=upload_result["secure_url"],
            bytes=upload_result.get("bytes"),
            format=upload_result.get("format"),
            version=upload_result.get("version"),
        )
