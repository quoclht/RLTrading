from absl import app, flags
from agents import train
from market_simulator.core.data_feed import fetch_data
flags.DEFINE_string(
    "task",
    None,
    'Tasks: \n \
                        + Update data("update_data"). \n \
                        + Train agent("rebalance").',
)

FLAGS = flags.FLAGS


def main(_):
    input_task = FLAGS.task
    if input_task == "update_data":
        fetch_data.update(symbols=['BTCUSDT', 'ETHUSDT'], with_zip=True)
    elif input_task == "train":
        train.main()


if __name__ == "__main__":
    app.run(main)
