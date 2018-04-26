pip install ./src/matkirpack/. --upgrade
pip install opencv-python
pip install tqdm
if [ ! -d "../kvasir-dataset-v2" ]; then
wget http://datasets.simula.no/kvasir/data/kvasir-dataset-v2.zip
unzip -o kvasir-dataset-v2.zip -d ../
rm kvasir-dataset-v2.zip
mkdir ../kvasir-dataset-v2/blanding
cp -r ../kvasir-dataset-v2/polyps/. ../kvasir-dataset-v2/blanding
cp -r ../kvasir-dataset-v2/normal-cecum/. ../kvasir-dataset-v2/blanding
python ../Master_programs/src/sorter/sorter.py -make 
fi
