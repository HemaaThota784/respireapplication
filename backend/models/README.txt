Copy these files into this folder (names exactly as shown):

AUDIO  (from your RespireAI_main_s77_Results folder)
  respire_logits.keras   (or respireai_main_s77.keras - either name is found automatically)
  norm_stats.json        <- REQUIRED (normalizer mean/std saved in training)
  preproc_config.json    <- saved in training
  meta.json              <- contains the calibrated "temperature"

X-RAY
  xray_pneumonia_model.keras   (rename from "xray_pneumonia_model.keras (1)" if needed)
