运行命令:
###
python transcribe_whisper.py --output asr.jsonl --model "/mnt/diskhd/Backup/DownloadModel/whisper-large-v3/" --input Mars2Dataset/MAC_QA.jsonl --output asr.jsonl 
###
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_mac.py --track mac   --input Mars2Dataset/MAC_QA.jsonl --output mac_result.jsonl  --videos Mars2Dataset/mars2_videos   --asr asr.jsonl --tensor-parallel-size 4   --batch-size 32 --model /mnt/diskhd/Backup/DownloadModel/Qwen3.5-9B/
