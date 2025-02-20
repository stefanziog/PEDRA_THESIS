import tensorflow as tf
import os

log_dir = r"C:\Users\Autostudents\PEDRA\models\trained\Indoor\Blocks\Imagenet\e2e\drone0\return_plot"

log_files = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f]

if not log_files:
    print("❌ No TensorBoard log files found in:", log_dir)
else:
    print("✅ Found log files:", log_files)

    for file in log_files:
        print(f"Checking file: {file}")
        try:
            for e in tf.compat.v1.train.summary_iterator(os.path.join(log_dir, file)):
                print("✅ Log file contains data!")
                break
        except Exception as ex:
            print(f"❌ Error reading {file}: {ex}")
