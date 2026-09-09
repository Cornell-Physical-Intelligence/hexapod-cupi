# Local inspection dependencies

Unmodified files copied from the already installed viewer dependencies:

- Three.js 0.169.0, MIT; license in `three/LICENSE`.
- urdf-loader 0.12.7, Apache-2.0; license in `urdf-loader/LICENSE`.

These pinned modules let `inspection_v2.html` run without Wi-Fi/CDNs. All
model files are loaded from this repository. ColladaLoader/TGALoader are
transitive static imports of URDFLoader; this model uses STL meshes only.
Upstream sources: https://github.com/mrdoob/three.js and
https://github.com/gkjohnson/urdf-loaders .
