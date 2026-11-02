import csv, logging
from datetime import datetime
from common.config import LOG_DIR

def setup_logger(name):
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger(name); logger.setLevel(logging.INFO); logger.handlers.clear()
    handler=logging.FileHandler(LOG_DIR/f"{name}.log",encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    csv_path=LOG_DIR/f"{name}.csv"
    if not csv_path.exists():
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp","Game","Score","Epsilon","Steps","Total Reward","Replay Saved"])
    return logger,csv_path

def log_game(logger,path,game,score,epsilon,steps,total_reward,replay_saved):
    stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([stamp,game,score,round(epsilon,2),steps,round(total_reward,2),replay_saved])
    logger.info("Game %s | Score %s | Epsilon %.2f | Steps %s | Reward %.2f | Replay %s",
                game,score,epsilon,steps,total_reward,replay_saved)
