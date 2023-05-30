module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/yolov5
python3 detect.py --weights runs/train/exp21/weights/best.pt --source /gpfs/gpfs0/robert.jarolim/data/gregor/images/test --imgsz 1024 --save-txt
