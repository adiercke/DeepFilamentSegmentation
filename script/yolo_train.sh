module load python/pytorch-1.6.0
cd /beegfs/home/robert.jarolim/projects/yolov5
python3 train.py --imgsz 1024 --batch 4 --epochs 100 --data /beegfs/home/robert.jarolim/projects/yolov5/filament.yaml --weights yolov5l.pt --patience 10
