// Shared by the chat page and the memory viewer.
//
// Times arrive from the server in its timezone (UTC). Every <time datetime="...">
// is rewritten in this browser's local time: data-format="day" gives "Fri 21:57",
// "date" gives "Fri, Oct 16", "datetime" gives "Fri, Oct 16 21:57", and the default
// is "21:57".
//
// Buttons with data-confirm ask before submitting (forgetting is final).

function localTimes(root = document) {
  for (const time of root.querySelectorAll("time[datetime]")) {
    const moment = new Date(time.dateTime);
    if (Number.isNaN(moment.getTime())) continue;
    const clock = moment.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
    const weekday = moment.toLocaleDateString("en", { weekday: "short" });
    const date = moment.toLocaleDateString("en", { weekday: "short", month: "short", day: "numeric" });
    time.textContent = {
      day: `${weekday} ${clock}`,
      date,
      datetime: `${date} ${clock}`,
    }[time.dataset.format] || clock;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  localTimes();
  document.addEventListener("submit", (event) => {
    const button = event.submitter;
    const question = button?.dataset.confirm;
    if (question && !window.confirm(question)) event.preventDefault();
  });
});
