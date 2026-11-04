# Third-party code and data

- Runtime reference math originated from Holiday Robotics FlashDexRetarget;
  see `dex_track/tasks/reference/LICENSE` and `data_pipeline/FLASHDEX_LICENSE`.
- The bundled C solver is based on wuji-technology/TopoRetarget with XHand and
  mesh self-collision adaptations. Its license/notices are in `data_pipeline/solver/`.
- Audited MuJoCo-Warp collision files retain their Apache-2.0 headers and license
  in `runtime_patches/`. Local modifications implement pair-local collision
  coordinates and invalid-mesh-ID handling as recorded in its manifest.
- The packed resampling implementation is extracted from EGalahad/any4hdmi;
  its exact source fingerprint is in `data_pipeline/resampling-provenance.json`.
- OakInk2 and MANO assets/data retain their source terms. Licensed raw human
  models and recordings are not included in this source repository.
