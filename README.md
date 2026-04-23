viewtomo

Automated ViewTomo Alignment and Reconstruction

viewtomo is a Python-based pipeline designed to streamline the processing of "view magnification" datasets using AreTomo2 (default) or IMOD/Etomo.

Installation

Ensure you have IMOD and AreTomo2 installed and available in your PATH.

git clone [https://github.com/vojtaprazak/viewtomo.git](https://github.com/yourusername/viewtomo.git)

cd viewtomo
pip install -e .


Usage

Alignment & Reconstruction

Align one or more MRC stacks using the default AreTomo2 engine:

viewtomo_align.py stack1.mrc stack2.mrc


For IMOD-based patch tracking:

viewtomo_align.py stack.mrc --engine imod
