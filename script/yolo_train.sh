module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/yolov5
python3 train.py --imgsz 1024 --batch 8 --epochs 50 --data /beegfs/home/robert.jarolim/projects/yolov5/gregor.yaml --weights yolov5l.pt
