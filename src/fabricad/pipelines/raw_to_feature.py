#standard libaries


#3rd party libaries
from pathlib import Path
import shutil
import threading
import logging
from tqdm import tqdm
import pandas as pd
import json

#local libaries
from fabricad.constants import PATHS
from ezstep.conversions import CAD_Converter
from ezstep.conversions import logger as ezstep_logger
# suppress ezstep logger
ezstep_logger.setLevel(logging.WARNING)


# set up logger
logging_level = logging.WARNING
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class Raw_to_Feature:

    _instance = None

    def __new__(cls):
        """
        Create a new instance of the class if one does not already exist.

        This method ensures that only one instance of the pipeline class can exist
        (Singleton pattern). If an instance already exists, it returns the existing
        instance. Otherwise, it creates a new instance and returns it.

        Returns:
            cls: The single instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        #get generator for all dirs in the raw data folder
        logger.debug("Init Pipeline..")
        self._path_generator = PATHS.DATA_RAW.iterdir()

        self._targets= []
        self._known_classes = []


    def _get_next_path(self) -> None:
        """
        Retrieve the path to the next step file from the primary data set.

        This method uses a step path generator to obtain the path to the next file
        that needs to be processed. It updates the instance variable `_sample_to_process`
        with the path to this file.

        Returns
        -------
        None

        Raises
        ------
        StopIteration: If the step path generator has no more files to process.
        """
        # get path to next step file
        try:
            # Dein Code mit next(self._path_generator)
            self._sample_to_process = next(self._path_generator)
        except StopIteration as e:
            logger.error(f"StopIteration caught: {e}")
            raise e  # Um sicherzustellen, dass der Fehler korrekt weitergegeben wird
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise e

        self._file_id = self._sample_to_process.stem
        self._target_sample_dir = PATHS.DATA_FEATURE / self._sample_to_process.stem

    def _to_vecset(self):
        step_file = self._sample_to_process / f"geometry_{self._file_id}.STEP"
        logger.debug(f"Converting STEP file {step_file} to vecset")

        vs_target_file = self._target_sample_dir / "features/vecset.npy"

        if vs_target_file.exists():
            logger.debug(f"Vecset file {vs_target_file} already exists")
            return None
        
        vs_target_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            converter = CAD_Converter.from_step(step_file)
            converter.to_vecset(vs_target_file)
        except Exception as e:
            logger.error(f"Error converting STEP to vecset: {e}", exc_info=True)
            return None
        



    # def _convert_step_to_voxel(self):
    #     logger.debug("Start to convert STEP file to VOXEL. Start with STEP to STL..")
    #     step_file = self._file_to_process / f"geometry_{self._file_id}.STEP"
        
    #     interim_stl_file = PATHS.DATA_INTERIM / f"{self._file_id}/stl/{self._file_id}.stl"
    #     voxel_file = PATHS.DATA_FEATURE / f"{self._file_id}/voxel/{self._file_id}.npz"

    #     logger.debug(f"interim STL file: {interim_stl_file}")
    #     logger.debug(f"Voxel file: {voxel_file}")

    #     if voxel_file.exists():
    #         logger.debug(f"File {voxel_file} already exists")
    #         return None

    #     try:
    #         stl = Converter.convertStepToStl(step_file, interim_stl_file)
    #         voxel = Converter.stl_to_voxel(stl.converted_file, voxel_file,  128)
    #     except Exception as e:
    #         logger.error(f"Error: {e}", exc_info=True)
    #         #clean up
    #         if interim_stl_file.parent.exists():
    #             shutil.rmtree(interim_stl_file.parent)
    #         if voxel_file.exists():
    #             shutil.rmtree(voxel_file.parent)
    #         return None
        
    def _convert_metadata(self):
        self._metadata = pd.read_csv(self._sample_to_process / f"plan_metadata.csv", sep=";")

    def _convert_productionplan_data(self):
        self._production_plan = pd.read_csv(self._sample_to_process / f"plan.csv", sep=";").drop(0)

    def _convert_substep_informations(self):
        self._features = pd.read_csv(self._sample_to_process / "interim/substeps/features.csv", sep=";")

    def _extract_targets(self):
        self._sequence = self._production_plan["Schritt"].values.tolist()
        self._turned = "drehen" in self._sequence
        self._total_mill_features = self._get_total_of_features_for_step_type("fräsen")
        self._total_drill_features = self._get_total_of_features_for_step_type("bohren")
        self._total_welding_features = self._get_total_of_features_for_step_type("schweißen")
        self._total_grinding_features = self._get_total_of_features_for_step_type("schleifen")     
        self._total_features = len(self._features)

        self._targetDict = {"turned" : self._turned,
                           "total_mill_feats" : self._total_mill_features,
                           "total_drill_feats" : self._total_drill_features,
                           "total_weld_feats" : self._total_welding_features,
                           "total_grind_feats" : self._total_grinding_features,
                           "total_feats" : self._total_features,
                           "sequence" : self._sequence}

        #TODO cost and time functions - maby zdirs..
        pass

    def _convert_table_data(self):
        logger.debug("start to convert table data from csv files from the generated Data")
        self._convert_metadata()
        self._convert_productionplan_data()
        self._convert_substep_informations()

        self._extract_targets()
        self._save_data()

    def _save_data(self):
        logger.debug("save table data to files")
        path_to_table_data = self._target_sample_dir / "production_plan"
        path_to_prediction_targets = self._target_sample_dir / "prediction_targets"


        path_to_table_data.mkdir(parents=True, exist_ok=True)
        path_to_prediction_targets.mkdir(parents=True, exist_ok=True)

        self._features.to_json(path_to_table_data / "features.json")
        self._production_plan.to_json(path_to_table_data / "production_plan.json")

        self._metadata.to_json(self._target_sample_dir / "metadata.json")

        with open((path_to_prediction_targets / "targets.json").as_posix(), "w") as outfile: 
             json.dump(self._targetDict, outfile)
        logger.debug("Done")


    def _get_total_of_features_for_step_type(self, step : str) -> int:
        logger.debug(f"get total amount of features for step: {step}")
        filtered_step = self._production_plan[self._production_plan["Schritt"] == step]
        
        if len(filtered_step==0):
            return 0
        else:
            step_ids = filtered_step["Nr."].values
            features = self._features[self._features["Arbeitsschritt"].isin(step_ids)]
            return len(features)
        
    def _check__data_integrity(self):
            logger.debug("check data integrity")
            plan_file = self._target_sample_dir / "production_plan/production_plan.json"
            feature_file = self._target_sample_dir / "production_plan/features.json"
            prediction_file = self._target_sample_dir / "prediction_targets/targets.json"
            vecset_file = self._target_sample_dir / f"features/vecset.npy"

            if plan_file.exists()==False:
                logger.info(f"File {plan_file} does not exist")
                return False
            if feature_file.exists()==False:
                logger.info(f"File {feature_file} does not exist")
                return False
            if prediction_file.exists()==False:
                logger.info(f"File {prediction_file} does not exist")
                return False
            if vecset_file.exists()==False:
                logger.info(f"File {vecset_file} does not exist")
                return False

            return True

    def run(self):
        """
        Execute the entire pipeline.
        """

        # process all step files
        logger.info("Starting processing of step files.")
        progress_bar = tqdm(desc="Files processed: ",total=len(list(PATHS.DATA_RAW.iterdir())), ncols=100)

        try:
            while True:
                try:
                    self._get_next_path()

                    # skip if file already processed 
                    if self._target_sample_dir.exists() and self._check__data_integrity():
                        logger.info(f"Sample already exists: {self._target_sample_dir.stem}")
                        progress_bar.update(1)
                        continue

                    logger.debug(f"Processing {self._sample_to_process}")
                except StopIteration:
                    logger.info("All step files processed")
                    break

                try:
                    vecset_thread = threading.Thread(target=self._to_vecset)
                    table_thread = threading.Thread(target=self._convert_table_data)

                    logger.debug("Starting threads")
                    vecset_thread.start()
                    table_thread.start()


                    vecset_thread.join()
                    table_thread.join()

                    logger.debug("Threads finished")

                    if self._check__data_integrity()==False:
                        logger.warning(f"remove sample due to missing files: {self._target_sample_dir}")
                        shutil.rmtree(self._target_sample_dir)


                    progress_bar.update(1)

                except Exception as e:
                    logger.error(f"Error processing {self._sample_to_process}: {e}", exc_info=True)
                    continue
        finally:
            pass

if __name__ == "__main__":
    pipeline = Raw_to_Feature()
    pipeline.run()
