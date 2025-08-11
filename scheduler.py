# scheduler.py
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import subprocess

TZ = pytz.timezone("America/Los_Angeles")

def run_scan():
    # Call your scan module
    subprocess.run([ "./.venv/bin/python", "scan.py" ], check=False)

if __name__ == "__main__":
    sched = BlockingScheduler(timezone=TZ)
    # Weekdays at 13:35 PT
    sched.add_job(run_scan, CronTrigger(day_of_week="mon-fri", hour=13, minute=35))
    print("Scheduler started (Mon–Fri 13:35 PT). Ctrl-C to stop.")
    sched.start()
