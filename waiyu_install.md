# Set up conda environment to jupyter notebook 
conda install nb_conda
conda install ipykernel
python -m ipykernel install --user --name amodal_detectron2 --display-name "Python (myenv)"

# build environment with python 3.6.5 ( compatible with nb_coda and opencv)

# some command on remote server 
ssh -L 8000:localhost:8888 -L 16006:localhost:6006 waiyu@vision2.idav.ucdavis.edu

scp -r waiyu@vision2.idav.ucdavis.edu:/home/waiyu/amodal_detectron2 /Users/wylam/Documents/Courses/Winter2020/Amodal/ORCNN
