PYTHONPATH=/home/kolesov93/study/fairseq python3.6 main.py --use-fbank --wanted-words ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki --dev-every-batches 2048 --batch-size 16 --lr-drop 8 --lr 0.06 --model res8 /home/kolesov93/study/datasets/lt_data test_runs/baseline
PYTHONPATH=/home/kolesov93/study/fairseq python3.6 main.py --use-fbank --dev-every-batches 512 --batch-size 64 --lr-drop 7 --lr 0.6 --model res8 --self-pretrain /home/kolesov93/study/datasets/lt_pretrain test_runs/self_pretrain