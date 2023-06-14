module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/yolov5
python3 val.py --weights '/beegfs/home/robert.jarolim/projects/yolov5/runs/train/exp16/weights/last.pt' --imgsz 1024 --data /beegfs/home/robert.jarolim/projects/yolov5/gregor.yaml
