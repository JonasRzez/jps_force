python ini_mc_large_i.py
python mc_l_1.py
wait

cd texfiles/16_09_2021_videos/testvideo_



rm *

cd ../../..

python order_plot.py

cd texfiles/16_09_2021_videos/testvideo_



ffmpeg -f image2 -r 5 -i %03d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p  i23_5fps.mp4

open i23_5fps.mp4
