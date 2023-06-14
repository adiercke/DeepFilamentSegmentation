module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/yolov5
python3 detect.py --weights /beegfs/home/robert.jarolim/projects/yolov5_stash/runs/train/large/weights/best.pt --source /gpfs/gpfs0/robert.jarolim/data/filament/gong_img --imgsz 1024 --save-txt
python3 detect.py --weights /beegfs/home/robert.jarolim/projects/yolov5_stash/runs/train/large/weights/best.pt --source /gpfs/gpfs0/robert.jarolim/data/filament/yolov5_data/images/test --imgsz 1024 --save-txt