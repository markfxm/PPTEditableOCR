# Third-Party Notices

## Segment Anything 2 (SAM 2)

- Source: https://github.com/facebookresearch/sam2
- Model: `sam2.1_hiera_tiny`
- Copyright: Meta Platforms, Inc. and affiliates
- Code and model license: Apache License 2.0
- CC-Torch components: BSD 3-Clause License

The application downloads the official model checkpoint only after the user confirms the download. The checkpoint is stored outside the installation directory under the current user's local application data directory.

The complete license texts and notices are available in the upstream SAM 2 repository linked above. Redistribution builds must retain the applicable Apache 2.0 and BSD 3-Clause notices.

## PyMatting

- Source: https://github.com/pymatting/pymatting
- Purpose: local alpha matting for extracted image edges
- License: MIT License

The desktop build bundles PyMatting when installed. If it is unavailable at runtime, the application keeps a conservative OpenCV alpha fallback and does not upload assets.
