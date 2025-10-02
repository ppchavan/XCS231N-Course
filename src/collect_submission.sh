#!/bin/bash
set -euo pipefail

pushd submission

CODE=(
	# Transformer implementation files
	"xcs231n/transformer_layers.py"
	"xcs231n/classifiers/transformer.py"
	"xcs231n/captioning_solver_transformer.py"

	# Self-Supervised Learning implementation files
	"xcs231n/simclr/contrastive_loss.py"
	"xcs231n/simclr/data_utils.py"
	"xcs231n/simclr/utils.py"
	"xcs231n/simclr/model.py"
)

NOTEBOOKS=(
	"Transformer_Captioning.ipynb"
	"Self_Supervised_Learning.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
ZIP_FILENAME="../a3.zip"

C_R="\033[31m"
C_G="\033[32m"
C_BLD="\033[1m"
C_E="\033[0m"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") -x "makepdf.py"


FULL_ZIP_PATH=$(realpath ${ZIP_FILENAME})
BASE_NAME=$(basename ${ZIP_FILENAME})
echo -e "### Done! Please submit ${BASE_NAME} to Gradescope. ###"
echo -e "### Full path: ${FULL_ZIP_PATH} ###"

popd