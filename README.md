ViewTomo: Automated Cryo-ET Alignment and Reconstruction

viewtomo is a Python-based pipeline designed to streamline the processing of "view magnification" Cryo-Electron Tomography (Cryo-ET) datasets. It provides a highly robust, automated interface for both GPU-accelerated alignment via AreTomo2 and traditional CPU patch-tracking workflows using IMOD/Etomo.

📖 Read our preprint on bioRxiv: Automated ViewTomo Alignment and Reconstruction

# 🚀 Quickstart (For the Impatient, a detailed guide is below)

If you are familiar with Python virtual environments and Git, you can get started in seconds:

1. Install

git clone [https://github.com/vojtaprazak/ViewTomo.git](https://github.com/vojtaprazak/ViewTomo.git)
cd ViewTomo
python3 -m venv viewtomo_venv
source viewtomo_venv/bin/activate
pip install -e .


2. Basic Usage

Run the pipeline on a single tilt series (uses AreTomo2 by default):

viewtomo_align.py ts01.mrc


Or process an entire folder of tilt series automatically:
example/
├── ts01.mrc
└── ts02.mrc

viewtomo_align.py ts*.mrc

This creates
example/
├── ts01
└── ts02




🧠 How It Works (The Procedure)

viewtomo is designed to be an automated pipeline that still leaves you with full control over the final data. The script performs the following sequential steps for every .mrc stack provided:

Automated deterministic histogram-based Masking: Identifies and inpaints obscured regions (e.g., thick ice, grid bars) and vacuum using Beer-Lambert and polynomial traces. This helps with alignment stability on lamella.

Tilt Series Alignment: Aligns the masked stack using either markerless AreTomo2 (default) or IMOD patch-tracking.

Etomo Project Creation: Seamlessly builds a native IMOD/Etomo project, translating AreTomo2 coordinates into IMOD-compatible directives.

Tomogram Reconstruction: Calculates positioning, pitch, and generates the final reconstructed tomogram in Etomo.

Crucial Feature: Because viewtomo builds a standard IMOD project in the background, you can always open the resulting .edf file in the Etomo GUI. If you are unhappy with the automated reconstruction, simply open your_dataset.edf in Etomo and modify, re-align, or re-reconstruct any part of it exactly as you would normally!

# 🐢 Detailed Guide

If you are new to using GitHub, Python packages, or the command line, follow these steps.

Prerequisites

Before installing viewtomo, ensure your system has the following software installed and available in your terminal's PATH:

IMOD

AreTomo2

## Step 1: Download the Code

You need to copy the code from GitHub to your local computer or cluster. In your terminal, run:

git clone [https://github.com/vojtaprazak/ViewTomo.git](https://github.com/vojtaprazak/ViewTomo.git)
cd ViewTomo


### Optional Step 2: Create a Virtual Environment

A virtual environment is a self-contained folder that holds Python packages. It prevents viewtomo's dependencies from clashing with other Python software on your system.

### Create the environment
python3 -m venv viewtomo_venv

### Activate the environment (you must do this every time you open a new terminal)
source viewtomo_venv/bin/activate


(You will know it worked if your terminal prompt now starts with (viewtomo_venv)).

## Step 3: Install the Package

Now, use pip (Python's package installer) to install the code. The -e . flag stands for "editable", meaning if you update the code via git pull later, you won't need to reinstall it.

pip install -e .


## Step 4: Run the Script

As long as your virtual environment is active, the viewtomo_align.py command is now available anywhere on your system.

Navigate to the folder containing your data
cd /path/to/your/mrc/files/

Run the alignment script
viewtomo_align.py my_tilt_series.mrc


⚙️ Advanced Usage & Parameters

You can customize the pipeline's behavior using command-line arguments.

Engine Selection & Geometry:

--engine imod: Switch from AreTomo2 to classical IMOD patch tracking.

--align_nm 150 / --final_nm 300: Set target thicknesses for alignment and reconstruction (in nanometers).

--aretomo_binning 4 / --tomo_binning 4: Binning factors for the alignment pass and the final tomogram output.

--imagebinned 1: If your input .mrc files are already binned, specify the factor here so patch tracking sizes scale correctly.

--template lamella.adoc: Path to IMOD system template (.adoc). Uses default lamella.adoc in templates.

Masking Parameters:
If the automated masking is too harsh or too gentle, you can tune it:

--mask_low_cut 0.05: Lower values result in gentler masking of dark/obscured features.

--mask_high_cut 0.05: Lower values result in gentler masking of bright/vacuum features.

--mask_dilation 5: How far the mask expands (in pixels) into the surrounding area.

--wiggle 1.0: Relaxation factor. Higher values make the mask less stringent at high tilt angles.

--debug: Saves a .png and .json file detailing the physics fits—highly recommended if you need to troubleshoot why a specific tilt series isn't masking properly!

Example of an advanced run:

viewtomo_align.py *.mrc --engine imod --tomo_binning 8 --mask_low_cut 0.02 --debug
