####################
# STUDENT MAKEFILE #
####################
# Possible targets:
# `make without_solutions`
#   Creates the A1.pdf handout.  Removes auxiliary latex files after
#   compilation.  This creates the handout by creating a temporary copy of all
#   files, removing solution code, then deleting the temporary directory.
# `make clean`
#   Removes assignment handouts, both with and without solutions.

# If you want to edit your latex live (i.e. WYSIWYG), try running the following
# command in your tex/ directory:
#
# $ latexmk -pvc -jobname="A1_Solutions"
#
# (Of course, substituting the "A1" for the assignment root document)
#

SHELL = /bin/sh

THIS_ASSIGNMENT = A2

TEX_DEPENDENCIES = $(shell find tex -type f) points.json meta.json

.DEFAULT_GOAL := submission.pdf

.PHONY: clean

without_solutions: $(THIS_ASSIGNMENT).pdf

$(THIS_ASSIGNMENT).pdf: $(TEX_DEPENDENCIES)
	# 1. Make and move into a temp folder where the tex files can be modified (solutions removed)
	# 2. Move into the temp directory and remove solution code.
	# 3. Make the latex document and clean it up (if it compiles correctly)
	# 4. Withdraw and remove the temp directory
	mkdir -p ./temp_tex_no_solutions
	cp -a ./tex/* ./temp_tex_no_solutions/
	cd ./temp_tex_no_solutions ; \
	find "./" -name "*.tex" -exec sed -i -e '/.*START CODE HERE.*/,/.*END CODE HERE.*/{//!d;}' {} + ; \
	find "./" -name "*.tex" -exec sed -i -e '/.*BEGIN_HIDE.*/,/.*END_HIDE.*/{//!d;}' {} + ; \
	find "./" -name "*.tex" -exec sed -i -e '/SOLUTION ALERT/{N;d;}' {} + ; \
	latexmk -jobname="$(THIS_ASSIGNMENT)" && latexmk -jobname="$(THIS_ASSIGNMENT)" -c ; \
	cd .. ; \
	rm -rf ./temp_tex_no_solutions

clean:
	cd ./tex; latexmk -jobname="$(THIS_ASSIGNMENT)" -C
	cd ./tex; latexmk -jobname="$(THIS_ASSIGNMENT)_Solutions" -C
	rm -f $(THIS_ASSIGNMENT)*
	rm -rf ./temp_tex_no_solutions