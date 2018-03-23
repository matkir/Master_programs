pip install ./src/matkirpack/. --upgrade

if [ ! -d "kvasir-dataset-v2" ]; then
wget http://datasets.simula.no/kvasir/data/kvasir-dataset-v2.zip
unzip -o kvasir-dataset-v2.zip
rm kvasir-dataset-v2.zip
fi

