import json
import requests
from datetime import datetime, timedelta, date
from ics import Calendar, Event

JSON_URL = "https://windsorislamicassociation.github.io/WIAPrayer.json"


def get_prayer_record(target_date: date):
    resp = requests.get(JSON_URL)
    data = resp.json()
    for rec in data:
        if rec["Month"] == target_date.month and rec["Day"] == target_date.day:
            return rec
    raise ValueError(f"Date {target_date} not found in prayer data")


def parse_time(target_date: date, time_str: str, meridiem: str) -> datetime:
    return datetime.strptime(
        f"{target_date.isoformat()} {time_str} {meridiem}", "%Y-%m-%d %I:%M %p"
    )


def build_schedule(target_date: date) -> Calendar:
    today_rec = get_prayer_record(target_date)
    next_day = target_date + timedelta(days=1)
    next_rec = get_prayer_record(next_day)

    fajr_next = parse_time(next_day, next_rec["Fajr"], "AM")
    sunrise_next = parse_time(next_day, next_rec["Sunrise"], "AM")
    isha_today = parse_time(target_date, today_rec["Isha"], "PM")

    sleep_start = fajr_next - timedelta(hours=8)
    isha_plus = isha_today + timedelta(minutes=10)
    if sleep_start < isha_plus:
        sleep_start = isha_plus

    wake_time = sunrise_next - timedelta(minutes=20)

    c = Calendar()

    e_sleep = Event()
    e_sleep.name = "Sleep"
    e_sleep.begin = sleep_start
    e_sleep.end = wake_time
    c.events.add(e_sleep)

    e_fajr = Event()
    e_fajr.name = "Pray Fajr"
    e_fajr.begin = wake_time
    e_fajr.end = wake_time + timedelta(minutes=10)
    c.events.add(e_fajr)

    e_run = Event()
    e_run.name = "Post-Fajr Run"
    e_run.begin = e_fajr.end
    e_run.end = e_run.begin + timedelta(minutes=15)
    c.events.add(e_run)

    # Evening schedule
    dinner_end = sleep_start - timedelta(hours=2)
    dinner_start = dinner_end - timedelta(minutes=30)
    workout_start = dinner_start - timedelta(minutes=100)

    e_workout = Event()
    e_workout.name = "Evening Workout"
    e_workout.begin = workout_start
    e_workout.end = dinner_start
    c.events.add(e_workout)

    e_dinner = Event()
    e_dinner.name = "Dinner"
    e_dinner.begin = dinner_start
    e_dinner.end = dinner_end
    c.events.add(e_dinner)

    e_night = Event()
    e_night.name = "Night Routine"
    e_night.begin = dinner_end
    e_night.end = isha_today
    c.events.add(e_night)

    e_isha = Event()
    e_isha.name = "Isha Prayer"
    e_isha.begin = isha_today
    e_isha.end = isha_today + timedelta(minutes=10)
    c.events.add(e_isha)

    return c


def main():
    target_date = date(2025, 7, 14)
    cal = build_schedule(target_date)
    with open("schedule.ics", "w") as f:
        f.writelines(cal)
    print("ICS file written as schedule.ics")


if __name__ == "__main__":
    main()
