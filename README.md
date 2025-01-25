# Visual Odometry with Monocular Camera

This project implements a visual odometry pipeline using a monocular camera to estimate the 3D trajectory of a moving platform. The pipeline integrates feature extraction, matching, and motion estimation while handling real-world challenges such as scaling with speedometer data.

## Features

- **Feature Extraction and Matching**: Efficiently extract and match features between consecutive frames.
- **Motion Estimation**: Estimate relative camera motion using epipolar geometry and recover poses.
- **Trajectory Visualization**: Plot and compare estimated trajectories against ground truth in 3D.

## Dependencies

This project uses the following dependencies:

- Python 3.13.0
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Custom modules:
  - `Matching.matcher`
  - `Extraction.extractor`
  - `motion_estimation.estimator`
  - `utils` (config loading, dataset processing, image utilities, plotting)

To install the required Python libraries:
```bash
pip install -r requirements.txt
```

## Project Structure

- `Matching/`: Contains the feature matching implementation.
- `Extraction/`: Responsible for feature extraction from images.
- `motion_estimation/`: Includes motion estimation logic.
- `utils/`: Utility functions for dataset processing, plotting, and configuration management.
- `configs/`: Configuration files for the pipeline.
- `Results/`: Stores generated results such as trajectory plots.

## Usage

1. Download the AdverCity dataset from [this link](https://labs.cs.queensu.ca/quarrg/datasets/adver-city/).
2. Place the dataset in your desired directory.
3. Change the path to the dataset in the `main.py` file.
4. Update the `config.yaml` file in the `configs` folder to adjust parameters as needed.
5. Run the main script:
```bash
python main.py
```
5. Results, including trajectory and error plots, will be saved in the `Results` folder.

## Contact

For questions or collaboration, feel free to reach out:
- **Email**: moez.rashed@queensu.ca

---

Enjoy experimenting with visual odometry!
