"""drcool: analysis tools for dilution-refrigerator parasitic heat-load measurements.

This module supports characterizing the parasitic heat load on the mixing
chamber (MC) and still of a dilution refrigerator, using a mixing-chamber
heater sweep and a still-heater sweep. It is organized in the order the
functions are used during an experiment:

1. **Data handling** -- load each control program's CSV log, apply clock
   offsets if needed, and merge logs with different sample rates onto a
   common timeline (:func:`load_log`, :func:`merge_logs`,
   :func:`to_seconds`, :func:`heater_power`, plus :func:`load_leiden_log`,
   :func:`from_labview_time`, :func:`heater_power_from_current`,
   :func:`clean_leiden_csv`, :func:`read_leiden_header`,
   :func:`load_leiden_logs` for Leiden Cryogenics logs).
2. **Noise characterization** -- quantify the MC thermometer noise from
   quiet base-temperature data, check whether it is white, and measure the
   empirical uncertainty of fitted slopes vs. window length
   (:func:`detrend`, :func:`allan_deviation`, :func:`autocorrelation`,
   :func:`allan_slope`, :func:`resample_even`,
   :func:`effective_samples`, :func:`windowed_slopes`,
   :func:`slope_scatter`, :func:`slope_sigma_white`).
3. **Settling** -- fit the exponential approach to steady state after a
   heater step, set the dT/dt threshold from the accuracy target, choose a
   slope-fitting window, and decide when a point has settled
   (:func:`fit_approach`, :func:`slope_threshold`, :func:`choose_window`,
   :func:`is_settled`).
4. **Planning** -- estimate the cooling coefficient ``a`` from a first
   heater step and lay out heater powers evenly spaced in T^2
   (:func:`estimate_a`, :func:`heater_plan`, :func:`predicted_temperature`).
5. **Results** -- fit settled points to extract the MC parasitic load,
   compare against ideal heat-exchanger performance, check repeat points
   for drift, and extrapolate the still parasitic load
   (:func:`fit_cooling_power`, :func:`exchanger_ratio`,
   :func:`repeat_check`, :func:`fit_still_load`).
6. **Plotting** -- annotated figures of each fit, built with matplotlib
   (imported only when used): :func:`plot_cooling_fit`,
   :func:`plot_still_fit`, :func:`plot_allan`, :func:`plot_approach`.

Physical model
--------------
At fixed circulation, the MC cooling power follows (Pobell, eq. for the
dilution cycle)::

    Q_cool = ndot_3 * (95 T_MC^2 - 11 T_in^2)  [W, ndot in mol/s, T in K]

which reduces to ``84 ndot_3 T^2`` for perfect heat exchangers
(T_in = T_MC). In steady state the applied heater power plus the
parasitic load equals the cooling power, so the analysis fits::

    P_heater = a * T_MC^2 - Q_par

The intercept at T^2 = 0 gives ``-Q_par``, the effective parasitic load
(true heat leak plus heat-exchanger inefficiency). At base temperature
(P_heater = 0), ``T_base = sqrt(Q_par / a)``.

For the still, circulation scales with total still heat input::

    ndot = (P_still + Q_par_still) / L_eff

so extrapolating flow vs. still heater power to zero flow gives the still's
parasitic load.

Conventions
-----------
* Units are SI throughout: K, W, s, mol/s, J/mol. Convert for display
  only; the ``summary()`` methods of result classes print in mK, nW, uW.
* Time arrays are float seconds with any origin, unless a function says
  it takes datetimes.
* Thermometer noise and settling functions expect evenly or near-evenly
  sampled data; :func:`allan_deviation` requires even sampling.

Typical workflow
----------------
>>> import drcool as dc
>>> # 1. Noise from quiet base data (t in s, T in K)
>>> r = dc.detrend(t_base, T_base)
>>> taus, adev = dc.allan_deviation(r, dt=5.0)
>>> W_grid = np.arange(60, 900, 30.0)
>>> scatter = dc.slope_scatter(t_base, r, W_grid)
>>> # 2. First heater step: get tau and a
>>> step = dc.fit_approach(t_step, T_step, skip=60)
>>> a = dc.estimate_a(P_first, step.T_inf, T_base_mean)
>>> # 3. Settling parameters
>>> s = dc.slope_threshold(delta_t=0.3e-3, tau=step.tau)
>>> W = dc.choose_window(W_grid, scatter, s, step.tau)
>>> # 4. Plan the sweep
>>> powers, repeats = dc.heater_plan(a, T_base_mean, 3 * T_base_mean)
>>> # 5. After the sweep
>>> fit = dc.fit_cooling_power(T_settled, P_applied, sigma_t=20e-6)
>>> print(fit.summary())

References
----------
F. Pobell, *Matter and Methods at Low Temperatures*, 3rd ed. (Springer,
2007), dilution refrigeration chapter.
D. W. Allan, "Statistics of atomic frequency standards," Proc. IEEE 54,
221 (1966), for the Allan variance.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

FloatArray = NDArray[np.float64]

#: Ideal MC cooling coefficient for perfect heat exchangers, J/(mol K^2).
#: Cooling power is ``IDEAL_COEFF * ndot * T^2`` when T_in = T_MC.
IDEAL_COEFF = 84.0


def _arr(x: ArrayLike) -> FloatArray:
    """Convert array-like input to a float64 NumPy array.

    Parameters
    ----------
    x : array_like
        Scalar, list, tuple, or array.

    Returns
    -------
    ndarray of float64
    """
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# 1. Data handling
# ---------------------------------------------------------------------------


def heater_power(voltage: ArrayLike, current: ArrayLike) -> FloatArray:
    """Compute heater power from 4-wire voltage and current readings.

    Using measured voltage across the heater (sensed at the heater, not at
    the supply) and measured current gives the power actually dissipated
    in the heater, excluding lead losses. A current-source setpoint alone
    does not account for lead resistance or calibration offsets.

    Parameters
    ----------
    voltage : array_like
        Voltage across the heater, V, from the 4-wire sense leads.
    current : array_like
        Current through the heater, A.

    Returns
    -------
    ndarray
        Dissipated power ``V * I``, W.
    """
    return _arr(voltage) * _arr(current)


def _epoch_unit(values) -> str:
    """Infer the unit of Unix epoch times from their magnitude.

    Present-day epoch times are ~1.7e9 s, ~1.7e12 ms, ~1.7e15 us, or
    ~1.7e18 ns. Each unit spans a factor of 1000, so the boundaries are
    unambiguous for any date between roughly 1973 and 2286.

    Parameters
    ----------
    values : array_like
        Numeric epoch timestamps.

    Returns
    -------
    str
        One of ``"s"``, ``"ms"``, ``"us"``, ``"ns"``.

    Raises
    ------
    ValueError
        If the values do not fit any unit (e.g. LabVIEW times, or seconds
        since some other epoch).
    """
    v = float(np.nanmedian(np.abs(_arr(values))))
    for unit, lo in (("s", 1e8), ("ms", 1e11), ("us", 1e14), ("ns", 1e17)):
        if lo <= v < lo * 1e3:
            return unit
    raise ValueError(f"cannot infer an epoch unit for values near {v:.3g}")


def load_log(
    path: str,
    time_col: str,
    value_cols: dict[str, str] | list[str],
    clock_offset: float = 0.0,
    time_format: str | None = None,
    tz: str | None = None,
    epoch_unit: str | None = None,
    **read_csv_kwargs,
):
    """Load one control program's CSV log into a time-indexed DataFrame.

    Each program (temperature controller, gas handling, flowmeter, ...)
    usually writes its own log. This reads one of them, keeps only the
    columns of interest, optionally renames them so names stay unique
    across programs, applies a clock offset, and indexes by timestamp.

    Parameters
    ----------
    path : str or path-like
        CSV file to read.
    time_col : str
        Name of the timestamp column.
    value_cols : dict or list of str
        Columns to keep. A dict ``{"original": "new_name"}`` also renames
        them, which avoids collisions when merging (e.g. two programs that
        both call a column ``"T"``).
    clock_offset : float, optional
        Seconds to *add* to this log's timestamps to put it on the
        reference clock. Programs on a single PC share a clock, so this is
        normally 0.
    time_format : str, optional
        ``strftime`` format for the timestamps. Supplying it is faster and
        avoids ambiguous day/month parsing.
    tz : str, optional
        IANA time zone of naive (zone-less) timestamps in this log, e.g.
        ``"America/Los_Angeles"``. If given, timestamps are localized and
        converted to UTC so the log can be merged with UTC-based logs such
        as :func:`load_leiden_log`. Ambiguous or nonexistent local times
        around DST changes become NaT. Leave ``None`` for logs that are
        already zone-aware, or when every log uses the same naive clock.
        Not used with ``epoch_unit`` (epoch times are already UTC).
    epoch_unit : {"s", "ms", "us", "ns", "auto"}, optional
        Set when the time column holds numeric Unix epoch times (counted
        from 1970-01-01 00:00 UTC) instead of date strings. Gives the unit
        of the numbers; ``"auto"`` infers it from their magnitude. Results
        are zone-aware UTC, so they merge directly with
        :func:`load_leiden_log` output. A numeric time column without this
        argument raises ``ValueError``, because pandas would otherwise
        read it as nanoseconds and place the data in 1970.
    **read_csv_kwargs
        Passed through to :func:`pandas.read_csv` (e.g. ``sep``,
        ``skiprows``, ``comment``).

    Returns
    -------
    pandas.DataFrame
        Selected columns indexed by a sorted ``DatetimeIndex`` named
        ``"time"`` (UTC if ``tz`` or ``epoch_unit`` was given).

    Notes
    -----
    :func:`merge_logs` requires every log to use the same time convention:
    all zone-aware (UTC) or all naive. Mixing them raises an error.
    """
    import pandas as pd

    df = pd.read_csv(path, **read_csv_kwargs)
    raw = df[time_col]
    numeric = pd.api.types.is_numeric_dtype(raw)
    if epoch_unit is not None:
        if not numeric:
            raise ValueError(f"epoch_unit given but {time_col!r} is not numeric")
        if tz is not None:
            raise ValueError("epoch times are UTC; do not combine with tz")
        unit = _epoch_unit(raw) if epoch_unit == "auto" else epoch_unit
        t = pd.to_datetime(raw.astype(float), unit=unit, utc=True)
    elif numeric:
        raise ValueError(
            f"{time_col!r} is numeric; pass epoch_unit='s' (or 'ms', 'us', "
            "'ns', 'auto') for Unix epoch times, or use from_labview_time "
            "for LabVIEW timestamps"
        )
    else:
        t = pd.to_datetime(raw, format=time_format)
    if tz is not None:
        t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        t = t.dt.tz_convert("UTC")
    t = t + pd.to_timedelta(clock_offset, unit="s")
    if isinstance(value_cols, dict):
        out = df[list(value_cols)].rename(columns=value_cols)
    else:
        out = df[list(value_cols)]
    out = out.assign(time=t).set_index("time").sort_index()
    return out


def merge_logs(frames: list, tolerance_s: float | None = None):
    """Merge several time-indexed logs onto the timeline of the first one.

    For each timestamp in ``frames[0]``, takes the most recent earlier
    sample from every other log (``pandas.merge_asof`` with
    ``direction="backward"``). This combines logs with different sample
    rates without interpolating. Put your highest-rate or most important
    log (usually the MC thermometer) first.

    Parameters
    ----------
    frames : list of pandas.DataFrame
        Logs from :func:`load_log`, each indexed by ``"time"``. Column
        names must be unique across frames.
    tolerance_s : float, optional
        Maximum age, in seconds, of a matched sample. Older matches become
        NaN instead of carrying a stale value forward (e.g. when one
        program stopped logging). ``None`` allows any age.

    Returns
    -------
    pandas.DataFrame
        All columns, indexed by the first frame's timestamps.

    Notes
    -----
    All frames must share a time convention: all zone-aware or all naive.
    Use ``tz=`` in :func:`load_log` to make naive local-time logs
    zone-aware before merging them with :func:`load_leiden_log` output.
    """
    import pandas as pd

    def prep(f):
        # merge_asof requires identical datetime resolution (ns vs us) too.
        f = f.reset_index()
        f["time"] = f["time"].dt.as_unit("ns")
        return f

    tol = pd.Timedelta(seconds=tolerance_s) if tolerance_s is not None else None
    merged = prep(frames[0])
    for f in frames[1:]:
        merged = pd.merge_asof(
            merged, prep(f), on="time", direction="backward", tolerance=tol
        )
    return merged.set_index("time")


#: LabVIEW timestamp epoch: seconds are counted from 1904-01-01 00:00 UTC.
LABVIEW_EPOCH = "1904-01-01"


def from_labview_time(seconds: ArrayLike):
    """Convert LabVIEW timestamps to zone-aware UTC datetimes.

    LabVIEW (used by Leiden Cryogenics and many other control programs)
    stores time as float seconds since 1904-01-01 00:00 UTC. Unlike a
    local-time string, this is unambiguous across DST changes and keeps
    sub-second resolution.

    Parameters
    ----------
    seconds : array_like
        LabVIEW timestamps, s.

    Returns
    -------
    pandas.DatetimeIndex
        UTC timestamps.
    """
    import pandas as pd

    epoch = pd.Timestamp(LABVIEW_EPOCH, tz="UTC")
    # Round to 1 ms: removes float representation noise (3.8748603826e9 is
    # stored as ...382.5999999) without touching the logged 0.1 s resolution.
    dt = pd.to_timedelta(_arr(seconds), unit="s").round("ms")
    return pd.DatetimeIndex(epoch + dt)


def load_leiden_log(
    path: str,
    channels: dict[int, str] | None = None,
    heaters: dict[int, str] | None = None,
    local_tz: str | None = None,
    drop_unused: bool = True,
):
    """Load a Leiden Cryogenics temperature-controller CSV log.

    Accepts all three file variants the logger produces, detected
    automatically (see :func:`read_leiden_table`): the raw format with its
    multi-line tab-separated header, headerless weekly continuation files,
    and the cleaned format written by :func:`clean_leiden_csv`. Columns
    follow the cleaned schema: ``date_string`` (local time, 1 s
    resolution), ``date_number`` (LabVIEW timestamp, UTC), ``res_ch0..9``
    (sensor resistance, ohm), ``temp_ch0_mk..temp_ch9_mk`` (temperature,
    mK), and ``heater_current_0..3``.

    Processing:

    * Time comes from ``date_number`` (UTC, 0.1 s resolution, no DST
      ambiguity) rather than ``date_string``.
    * Temperatures are converted from mK to K, matching the rest of this
      module.
    * ``Inf``/``-Inf`` (open or out-of-range channels) become NaN.
    * Channels whose resistance reads exactly 0 are treated as unused:
      their resistance and temperature become NaN, and if
      ``drop_unused`` is True, channels unused for the whole file are
      dropped.

    Parameters
    ----------
    path : str or path-like
        CSV file to read.
    channels : dict of {int: str}, optional
        Thermometer channels to keep and their names, e.g.
        ``{2: "mc"}`` gives columns ``T_mc`` (K) and ``R_mc`` (ohm). If
        omitted, all channels are kept as ``T_ch<n>`` / ``R_ch<n>``.
    heaters : dict of {int: str}, optional
        Heater channels to keep and their names, e.g. ``{0: "mc_heater"}``
        gives a column ``I_mc_heater``. If omitted, all are kept as
        ``I_heater<n>``. Values are passed through unconverted: check the
        controller's current units before computing power.
    local_tz : str, optional
        IANA zone to convert the index to (e.g. ``"America/Los_Angeles"``)
        for readability. Merging works the same either way, as long as all
        logs are zone-aware.
    drop_unused : bool, optional
        When ``channels`` is omitted, drop thermometer channels that are
        unused for the entire file (default True). Explicitly requested
        channels are always kept.

    Returns
    -------
    pandas.DataFrame
        Indexed by a zone-aware ``DatetimeIndex`` named ``"time"``.
    """
    import pandas as pd

    df = read_leiden_table(path)
    t = from_labview_time(df["date_number"])
    if local_tz is not None:
        t = t.tz_convert(local_tz)

    chans = channels if channels is not None else {n: f"ch{n}" for n in range(10)}
    heats = heaters if heaters is not None else {n: f"heater{n}" for n in range(4)}

    out = {}
    for n, name in chans.items():
        r = df[f"res_ch{n}"].astype(float).replace([np.inf, -np.inf], np.nan)
        temp = df[f"temp_ch{n}_mk"].astype(float).replace([np.inf, -np.inf], np.nan)
        unused = r == 0
        r, temp = r.mask(unused), temp.mask(unused) * 1e-3
        if drop_unused and unused.all() and channels is None:
            continue
        out[f"T_{name}"] = temp.to_numpy()
        out[f"R_{name}"] = r.to_numpy()
    for n, name in heats.items():
        out[f"I_{name}"] = df[f"heater_current_{n}"].astype(float).to_numpy()

    res = pd.DataFrame(out, index=pd.DatetimeIndex(t, name="time"))
    return res.sort_index()


#: Column names of the cleaned Leiden schema, in file order.
LEIDEN_COLUMNS = (
    ["date_string", "date_number"]
    + [f"res_ch{n}" for n in range(10)]
    + [f"temp_ch{n}_mk" for n in range(10)]
    + [f"heater_current_{n}" for n in range(4)]
)

_LEIDEN_DATA_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},")


@dataclass
class LeidenHeader:
    """Channel metadata from a raw Leiden log header.

    The raw header holds three tab-separated rows after a ``#`` line: column
    codes (``Date1, Date2, R0..R9, T0..T9, I0..I3``), channel names, and
    sensor calibrations. This collects the names and calibrations, which
    are useful for building the ``channels``/``heaters`` maps passed to
    :func:`load_leiden_log` and for recording which calibration was active.

    Attributes
    ----------
    channel_names : dict of {int: str}
        Thermometer channel number to its configured name (e.g.
        ``{2: "MXC (Low T)"}``). Channels with blank names are omitted.
    calibrations : dict of {int: str}
        Thermometer channel number to its calibration (e.g. ``"S0925"``,
        ``"RuO2 10k"``, ``"No Calibration"``).
    heater_names : dict of {int: str}
        Heater channel number to its name (e.g. ``{3: "MC"}``).
    """

    channel_names: dict[int, str] = field(default_factory=dict)
    calibrations: dict[int, str] = field(default_factory=dict)
    heater_names: dict[int, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a readable multi-line table of channels and heaters."""
        lines = ["Thermometers:"]
        for n in range(10):
            if n in self.channel_names or n in self.calibrations:
                name = self.channel_names.get(n, "")
                cal = self.calibrations.get(n, "")
                lines.append(f"  ch{n}: {name:<14} {cal}")
        lines.append("Heaters:")
        for n, name in sorted(self.heater_names.items()):
            lines.append(f"  heater{n}: {name}")
        return "\n".join(lines)


def _read_lines(path) -> list[str]:
    """Read a text file as lines, tolerating CRLF endings and stray bytes."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return f.read().splitlines()


def read_leiden_header(path) -> LeidenHeader | None:
    """Extract channel metadata from a raw Leiden log, if it has a header.

    Weekly continuation files have no header, so this returns ``None`` for
    them; take the metadata from the file written when the software was
    last started or reset. If a file contains several header blocks (the
    software was restarted while writing to it), the last one wins, since
    it describes the configuration of the most recent data.

    Parameters
    ----------
    path : str or path-like
        Raw Leiden log file.

    Returns
    -------
    LeidenHeader or None
        Metadata from the last header block, or ``None`` if none is found.
    """
    lines = _read_lines(path)
    header = None
    for i, line in enumerate(lines):
        if not line.startswith("Date1\t"):
            continue
        names = lines[i + 1].split("\t") if i + 1 < len(lines) else []
        cals = lines[i + 2].split("\t") if i + 2 < len(lines) else []
        names += [""] * (26 - len(names))
        cals += [""] * (26 - len(cals))
        header = LeidenHeader(
            channel_names={n: names[2 + n].strip() for n in range(10)},
            calibrations={n: cals[2 + n].strip() for n in range(10)},
            heater_names={n: names[22 + n].strip() for n in range(4)},
        )
        for d in (header.channel_names, header.calibrations, header.heater_names):
            for k in [k for k, v in d.items() if not v]:
                del d[k]
    return header


def read_leiden_table(path):
    """Read the data rows of any Leiden log variant into a DataFrame.

    Works on raw files (``#`` line, blank line, three tab-separated header
    rows, then comma-separated data), headerless weekly continuation
    files, cleaned files from :func:`clean_leiden_csv`, and files where
    the software appended a new header block mid-file after a restart.

    Rather than parsing headers, it keeps every line that looks like a data
    row (starts with a ``YYYY-MM-DD hh:mm:ss,`` timestamp) and has exactly
    26 fields, then assigns the cleaned column names. Header lines of every
    variant are skipped automatically, and a truncated final line (file
    still being written) is dropped instead of corrupting the parse.

    Parameters
    ----------
    path : str or path-like
        Leiden log file in any of the supported variants.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`LEIDEN_COLUMNS`, with numeric columns as float
        (``Inf``/``-Inf`` preserved; :func:`load_leiden_log` converts them
        to NaN). ``df.attrs["skipped_lines"]`` holds the number of
        timestamped lines dropped for having the wrong field count.

    Raises
    ------
    ValueError
        If the file contains no data rows.
    """
    import pandas as pd

    rows, skipped = [], 0
    for line in _read_lines(path):
        line = line.strip()
        if not _LEIDEN_DATA_LINE.match(line):
            continue
        if line.count(",") != len(LEIDEN_COLUMNS) - 1:
            skipped += 1
            continue
        rows.append(line)
    if not rows:
        raise ValueError(f"no Leiden data rows found in {path}")
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=LEIDEN_COLUMNS)
    df["date_string"] = df["date_string"].astype(str)
    num = LEIDEN_COLUMNS[1:]
    df[num] = df[num].astype(float)
    df.attrs["skipped_lines"] = skipped
    return df


def clean_leiden_csv(src, dst=None) -> str:
    """Convert a raw or headerless Leiden log to the cleaned CSV schema.

    Writes the cleaned header (``date_string, date_number, res_ch0, ...``)
    followed by the data rows exactly as logged, so values keep their
    original formatting and precision. Header blocks, blank lines, and
    truncated rows are dropped (see :func:`read_leiden_table`).

    Channel names and calibrations from the raw header are not part of the
    cleaned schema; use :func:`read_leiden_header` to keep them.

    Parameters
    ----------
    src : str or path-like
        Raw, headerless, or already-cleaned Leiden log.
    dst : str or path-like, optional
        Output file. If omitted, nothing is written.

    Returns
    -------
    str
        The cleaned CSV text, with ``\\n`` line endings.

    Raises
    ------
    ValueError
        If ``src`` contains no data rows.
    """
    rows = [
        line.strip()
        for line in _read_lines(src)
        if _LEIDEN_DATA_LINE.match(line.strip())
        and line.strip().count(",") == len(LEIDEN_COLUMNS) - 1
    ]
    if not rows:
        raise ValueError(f"no Leiden data rows found in {src}")
    text = ", ".join(LEIDEN_COLUMNS) + "\n" + "\n".join(rows) + "\n"
    if dst is not None:
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    return text


def load_leiden_logs(paths, **kwargs):
    """Load and concatenate several Leiden log files (e.g. weekly files).

    Each file is read with :func:`load_leiden_log`, so raw, headerless and
    cleaned files can be mixed. Results are sorted by time, and duplicate
    timestamps (overlap between consecutive files) are removed, keeping the
    first occurrence.

    Parameters
    ----------
    paths : iterable of str or path-like
        Files to load, in any order.
    **kwargs
        Passed to :func:`load_leiden_log` (``channels``, ``heaters``,
        ``local_tz``, ``drop_unused``). Pass explicit ``channels`` and
        ``heaters`` so every file yields the same columns; with automatic
        dropping, a channel unused in one week but not another would leave
        NaN gaps after concatenation.

    Returns
    -------
    pandas.DataFrame
        Combined log, indexed by time.
    """
    import pandas as pd

    frames = [load_leiden_log(p, **kwargs) for p in paths]
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]


def heater_power_from_current(current: ArrayLike, resistance: float) -> FloatArray:
    """Heater power from current and heater resistance, ``P = I^2 R``, in W.

    Use when only the heater current is logged (as in the Leiden logs).
    The result is only as good as ``resistance``: measure it 4-wire at the
    operating temperature, since heater resistance can change on cooldown,
    and confirm the logged current units (A vs. mA) first.

    Parameters
    ----------
    current : array_like
        Heater current, A.
    resistance : float
        Heater resistance at operating temperature, ohm.

    Returns
    -------
    ndarray
        Power dissipated in the heater, W.
    """
    return _arr(current) ** 2 * resistance


def to_seconds(times) -> FloatArray:
    """Convert datetimes to float seconds since the first sample.

    Most analysis functions here take time as float seconds. Use this on a
    merged log's index (or any datetime column) before analysis.

    Parameters
    ----------
    times : array_like of datetime-like
        Timestamps, e.g. a ``DatetimeIndex``.

    Returns
    -------
    ndarray
        Elapsed time in seconds; the first element is 0.
    """
    import pandas as pd

    t = pd.to_datetime(pd.Series(times))
    return (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 2. Noise characterization
# ---------------------------------------------------------------------------


def detrend(t: ArrayLike, y: ArrayLike, order: int = 1) -> FloatArray:
    """Remove a polynomial trend from a time series.

    Slow drift inflates the standard deviation and makes the
    autocorrelation large at all lags, hiding the actual noise. Subtracting
    a low-order polynomial fit isolates the fluctuations. Use ``order=1``
    for quiet base data; a higher order can absorb slow curvature but also
    removes genuine low-frequency noise, so keep it low.

    Parameters
    ----------
    t : array_like
        Time, s.
    y : array_like
        Signal, e.g. MC temperature in K.
    order : int, optional
        Polynomial order of the removed trend (default 1, linear).

    Returns
    -------
    ndarray
        Residuals ``y - fit(t)``, same units as ``y``.
    """
    t, y = _arr(t), _arr(y)
    tc = t - t.mean()
    return y - np.polyval(np.polyfit(tc, y, order), tc)


def allan_deviation(
    y: ArrayLike, dt: float, m: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray]:
    """Compute the overlapping Allan deviation of an evenly sampled series.

    The Allan deviation measures how much consecutive averages of length
    ``tau_avg`` differ. Unlike the standard deviation, it stays
    well-behaved in the presence of slow drift, and its log-log slope
    identifies which noise type dominates at each timescale:

    ========  ===================  ======================================
    Slope     Noise type           Implication
    ========  ===================  ======================================
    -1/2      white                averaging helps as 1/sqrt(N)
    0         flicker (1/f)        noise floor; longer averaging won't help
    +1/2      random walk          the quantity genuinely wanders
    +1        linear drift         still settling or drifting
    ========  ===================  ======================================

    Periodic disturbances (pulse tube, controller cycling) appear as bumps
    or dips near their period. The averaging time where the curve stops
    falling is the longest averaging that still reduces noise, which caps
    the useful slope-fitting window.

    Parameters
    ----------
    y : array_like
        Evenly sampled data, e.g. MC temperature in K. Detrending is not
        required: drift shows up as a +1 slope.
    dt : float
        Sample interval, s.
    m : array_like of int, optional
        Block sizes in samples. Defaults to ~40 log-spaced values up to
        ``N // 10`` so each point has at least ~10 independent blocks.
        Values with ``2 * m >= N`` are dropped.

    Returns
    -------
    tau_avg : ndarray
        Averaging times ``m * dt``, s.
    adev : ndarray
        Allan deviation at each ``tau_avg``, same units as ``y``.

    Notes
    -----
    Uses the overlapping estimator: block means are taken at every start
    index (via a cumulative sum), and the Allan variance is
    ``0.5 * mean((ybar[k+m] - ybar[k])**2)``. This uses the data more
    efficiently than non-overlapping blocks.
    """
    y = _arr(y)
    n = len(y)
    if m is None:
        m = np.unique(np.logspace(0, np.log10(max(n // 10, 1)), 40).astype(int))
    m = np.asarray(m, dtype=int)
    m = m[(m >= 1) & (2 * m < n)]
    c = np.concatenate([[0.0], np.cumsum(y)])
    adev = np.empty(len(m))
    for i, mi in enumerate(m):
        avg = (c[mi:] - c[:-mi]) / mi  # all overlapping block means
        d = avg[mi:] - avg[:-mi]  # adjacent-block differences
        adev[i] = np.sqrt(0.5 * np.mean(d**2))
    return m * dt, adev


#: Allan-deviation log-log slopes for common noise types.
ALLAN_NOISE_TYPES = {
    -0.5: "white",
    0.0: "flicker (1/f)",
    0.5: "random walk",
    1.0: "linear drift",
}


@dataclass
class AllanSlope:
    """Result of fitting a straight line to an Allan deviation in log-log.

    The model is ``log10(adev) = slope * log10(tau_avg) + intercept``.

    Attributes
    ----------
    slope : float
        Log-log slope over the fitted range.
    slope_err : float
        1-sigma uncertainty of ``slope``. Neighboring Allan points are
        correlated (they share data), so this is optimistic; treat
        differences smaller than ~2-3x this value as insignificant.
    intercept : float
        ``log10(adev)`` at ``tau_avg = 1 s``.
    tau_range : tuple of float
        Smallest and largest ``tau_avg`` actually used, s.
    n_points : int
        Number of Allan points in the fit.
    noise_type : str
        Noise type whose characteristic slope is closest (see
        :data:`ALLAN_NOISE_TYPES`). A slope between two reference values
        indicates a mixture.
    """

    slope: float
    slope_err: float
    intercept: float
    tau_range: tuple[float, float]
    n_points: int
    noise_type: str

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"slope = {self.slope:+.3f} +/- {self.slope_err:.3f} over "
            f"{self.tau_range[0]:.3g}-{self.tau_range[1]:.3g} s "
            f"({self.n_points} pts) -> closest: {self.noise_type}"
        )


def allan_slope(
    tau_avg: ArrayLike,
    adev: ArrayLike,
    tau_min: float | None = None,
    tau_max: float | None = None,
    n_samples: int | None = None,
    dt: float | None = None,
) -> AllanSlope:
    """Fit the log-log slope of an Allan deviation over a range of tau.

    Use on a region of the Allan curve that looks straight in log-log to
    identify the dominant noise type there (-1/2 white, 0 flicker, +1/2
    random walk, +1 linear drift).

    Parameters
    ----------
    tau_avg, adev : array_like
        Output of :func:`allan_deviation`.
    tau_min, tau_max : float, optional
        Inclusive range of ``tau_avg`` to fit, s. Defaults to the full
        range.
    n_samples, dt : int and float, optional
        Length and sample interval of the original series. If both are
        given, points are weighted by their statistical uncertainty: the
        relative error of an Allan deviation from ``K`` independent
        block differences is about ``1 / sqrt(2 K)``, with
        ``K = n_samples * dt / tau_avg - 1``, so long-tau points (few
        blocks) count less. Otherwise points are weighted equally and the
        uncertainty is scaled by the fit scatter.

    Returns
    -------
    AllanSlope
        Slope, uncertainty, intercept, range used, and closest noise type.

    Raises
    ------
    ValueError
        If fewer than 3 points fall in the range.
    """
    tau, ad = _arr(tau_avg), _arr(adev)
    sel = np.isfinite(ad) & (ad > 0)
    if tau_min is not None:
        sel &= tau >= tau_min
    if tau_max is not None:
        sel &= tau <= tau_max
    if sel.sum() < 3:
        raise ValueError("need at least 3 Allan points in the selected range")
    x, y = np.log10(tau[sel]), np.log10(ad[sel])
    sigma = None
    if n_samples is not None and dt is not None:
        k = np.maximum(n_samples * dt / tau[sel] - 1.0, 1.0)
        sigma = 1.0 / np.sqrt(2.0 * k) / np.log(10)  # error in log10(adev)
    m, b, cov, _ = _linfit(x, y, sigma)
    nearest = min(ALLAN_NOISE_TYPES, key=lambda r: abs(r - m))
    return AllanSlope(
        slope=m,
        slope_err=float(np.sqrt(cov[0, 0])),
        intercept=b,
        tau_range=(float(tau[sel].min()), float(tau[sel].max())),
        n_points=int(sel.sum()),
        noise_type=ALLAN_NOISE_TYPES[nearest],
    )


def resample_even(t: ArrayLike, y: ArrayLike, dt: float | None = None):
    """Interpolate a series onto an evenly spaced time grid.

    :func:`allan_deviation` assumes even sampling. Logged data often has
    jitter or occasional gaps; this puts it on a uniform grid by linear
    interpolation. Large gaps are bridged by straight lines, which
    suppress noise there, so split the data at gaps much longer than
    ``dt`` instead of resampling across them.

    Parameters
    ----------
    t : array_like
        Sample times, s (sorted).
    y : array_like
        Signal.
    dt : float, optional
        Grid spacing, s. Defaults to the median sample interval.

    Returns
    -------
    t_even : ndarray
        Evenly spaced times, s.
    y_even : ndarray
        Interpolated signal.
    dt : float
        Grid spacing used, s.
    """
    t, y = _arr(t), _arr(y)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if dt is None:
        dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    return grid, np.interp(grid, t, y), dt


def autocorrelation(y: ArrayLike, max_lag: int | None = None) -> FloatArray:
    """Compute the normalized autocorrelation of a series.

    ``rho(k)`` is the correlation between the series and itself shifted by
    ``k`` samples. For white noise, ``rho(k)`` is within about
    ``+/- 2/sqrt(N)`` of zero for all ``k != 0``. Correlated noise decays
    gradually over its correlation time; periodic noise oscillates at its
    period.

    Parameters
    ----------
    y : array_like
        Evenly sampled data. **Detrend first** (:func:`detrend`): residual
        drift makes ``rho`` large at all lags.
    max_lag : int, optional
        Largest lag to return, in samples. Defaults to ``N - 1``.

    Returns
    -------
    ndarray
        ``rho(k)`` for ``k = 0 .. max_lag``, with ``rho(0) = 1``.

    Notes
    -----
    Computed via zero-padded FFT, so it is fast for long records. Uses the
    biased estimator (normalized by ``N`` rather than ``N - k``), which is
    standard and keeps large-lag values from blowing up.
    """
    y = _arr(y)
    y = y - y.mean()
    n = len(y)
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return acf / acf[0]


def effective_samples(rho: ArrayLike, n: int) -> float:
    """Estimate the effective number of independent samples.

    Correlated samples carry less information than independent ones, so
    error formulas that assume white noise (like
    :func:`slope_sigma_white`) underestimate uncertainty. Substituting
    ``N_eff`` for ``N`` corrects for this to first order.

    Parameters
    ----------
    rho : array_like
        Autocorrelation from :func:`autocorrelation`, starting at lag 0.
    n : int
        Number of samples in the original series.

    Returns
    -------
    float
        ``N_eff = N / (1 + 2 * sum_k (1 - k/N) * rho(k))``. Equals ``N``
        for white noise and is smaller for positively correlated noise.

    Notes
    -----
    The sum stops at the first lag where ``rho <= 0``, a standard
    truncation that avoids summing estimation noise at large lags.
    """
    rho = _arr(rho)
    total = 0.0
    for k in range(1, len(rho)):
        if rho[k] <= 0:
            break
        total += (1 - k / n) * rho[k]
    return n / (1 + 2 * total)


def windowed_slopes(
    t: ArrayLike, y: ArrayLike, window: float, step: float | None = None
) -> tuple[FloatArray, FloatArray]:
    """Fit a straight line in successive time windows and return the slopes.

    A linear fit over a window averages down thermometer noise, giving a
    far better ``dT/dt`` estimate than point-to-point differences.

    Parameters
    ----------
    t : array_like
        Time, s (sorted).
    y : array_like
        Signal, e.g. MC temperature in K.
    window : float
        Window length, s.
    step : float, optional
        Spacing between window starts, s. Defaults to ``window``
        (non-overlapping). Use ``window / 2`` for half-overlapping windows.

    Returns
    -------
    centers : ndarray
        Window center times, s.
    slopes : ndarray
        Fitted slopes ``dy/dt`` in units of ``y`` per second (K/s for
        temperature). Windows with fewer than 3 samples are skipped.
    """
    t, y = _arr(t), _arr(y)
    step = window if step is None else step
    centers, slopes = [], []
    start = t[0]
    while start + window <= t[-1]:
        sel = (t >= start) & (t < start + window)
        if sel.sum() >= 3:
            ts = t[sel]
            slopes.append(np.polyfit(ts - ts.mean(), y[sel], 1)[0])
            centers.append(start + window / 2)
        start += step
    return np.array(centers), np.array(slopes)


def slope_scatter(t: ArrayLike, y: ArrayLike, windows: ArrayLike) -> FloatArray:
    """Measure the empirical slope uncertainty vs. window length.

    Fits slopes in non-overlapping windows of each length across quiet
    data (where the true slope is zero) and returns their standard
    deviation. This is the noise-limited uncertainty of a ``dT/dt``
    estimate, and because it is measured directly, it automatically
    includes any noise correlations: no white-noise assumption is needed.
    Feed the result to :func:`choose_window`.

    Parameters
    ----------
    t : array_like
        Time, s.
    y : array_like
        Quiet data, ideally detrended base-temperature data (K). Any
        residual drift adds a constant offset to the slopes and is not
        captured by the scatter, so detrend first.
    windows : array_like
        Window lengths to test, s.

    Returns
    -------
    ndarray
        Standard deviation of fitted slopes for each window, K/s. NaN
        where fewer than 2 windows fit in the data. For reliable values,
        use data spanning at least ~20 windows of the longest length.
    """
    out = []
    for w in _arr(windows):
        _, s = windowed_slopes(t, y, w)
        out.append(np.std(s, ddof=1) if len(s) > 1 else np.nan)
    return np.array(out)


def slope_sigma_white(sigma_t: float, n: int, window: float) -> float:
    """Slope uncertainty for a linear fit to white noise.

    For ``N`` evenly spaced points spanning a window ``W``, each with
    independent noise ``sigma_T``, the slope uncertainty is approximately
    ``sigma_T * sqrt(12 / N) / W``. Compare with :func:`slope_scatter`: if
    the empirical scatter is substantially larger, the noise is correlated
    (replace ``n`` with :func:`effective_samples` or rely on the empirical
    value).

    Parameters
    ----------
    sigma_t : float
        Per-sample noise standard deviation, K.
    n : int
        Number of samples in the window.
    window : float
        Window length, s.

    Returns
    -------
    float
        Slope standard deviation, K/s.
    """
    return sigma_t * np.sqrt(12.0 / n) / window


# ---------------------------------------------------------------------------
# 3. Settling
# ---------------------------------------------------------------------------


@dataclass
class SettleFit:
    """Result of fitting an exponential approach to steady state.

    The model is ``T(t) = T_inf + amplitude * exp(-t / tau)``, with ``t``
    measured from the start of the fitted data.

    Attributes
    ----------
    T_inf : float
        Extrapolated steady-state temperature, K.
    T_inf_err : float
        1-sigma uncertainty of ``T_inf``, K (from the fit covariance).
    tau : float
        Thermal time constant, s.
    tau_err : float
        1-sigma uncertainty of ``tau``, s.
    amplitude : float
        Remaining offset at the start of the fitted data, K. Negative when
        warming toward ``T_inf`` (heater stepped up).
    """

    T_inf: float
    T_inf_err: float
    tau: float
    tau_err: float
    amplitude: float

    def summary(self) -> str:
        """Return a one-line human-readable summary (mK and s)."""
        return (
            f"T_inf = {self.T_inf * 1e3:.4f} +/- {self.T_inf_err * 1e3:.4f} mK, "
            f"tau = {self.tau:.0f} +/- {self.tau_err:.0f} s"
        )


def _approach(t, t_inf, amp, tau):
    """Exponential approach model ``t_inf + amp * exp(-t / tau)``.

    Used internally by :func:`fit_approach` with ``scipy.optimize.curve_fit``.
    """
    return t_inf + amp * np.exp(-t / tau)


def fit_approach(t: ArrayLike, temp: ArrayLike, skip: float = 0.0) -> SettleFit:
    """Fit the exponential approach to steady state after a heater step.

    Serves two purposes: measuring the thermal time constant ``tau`` (from
    the first heater step, to set the slope threshold and window), and
    extrapolating the final temperature ``T_inf`` without waiting for full
    settling.

    Parameters
    ----------
    t : array_like
        Time, s, starting at (or after) the heater change. Any origin.
    temp : array_like
        MC temperature, K.
    skip : float, optional
        Seconds to discard after the first sample, excluding the fast
        initial transient that often follows a heater step before the
        slow exponential dominates. Default 0.

    Returns
    -------
    SettleFit
        Fitted ``T_inf``, ``tau``, amplitude, and uncertainties.

    Notes
    -----
    The model has a single time constant. If the system has several (e.g.
    fast MC response plus slow circulation re-equilibration), an early fit
    underestimates the final shift. Refit as data accumulates and trust
    ``T_inf`` only once it and ``tau`` stop drifting between refits;
    validate against a few fully settled points before relying on
    extrapolation.

    Raises
    ------
    RuntimeError
        If ``curve_fit`` fails to converge (e.g. too little data or no
        visible approach).
    """
    t, temp = _arr(t), _arr(temp)
    sel = t >= t[0] + skip
    t0 = t[sel] - t[sel][0]
    y = temp[sel]
    span = max(t0[-1], 1.0)
    p0 = (y[-1], y[0] - y[-1], span / 3)
    p, cov = curve_fit(_approach, t0, y, p0=p0, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    return SettleFit(p[0], err[0], abs(p[2]), err[2], p[1])


def slope_threshold(delta_t: float, tau: float) -> float:
    """Compute the dT/dt settling threshold from an accuracy target.

    For an exponential approach, the remaining offset from steady state is
    ``tau * |dT/dt|``. Requiring that offset to be below ``delta_t`` gives
    the threshold ``s = delta_t / tau``.

    Parameters
    ----------
    delta_t : float
        Acceptable per-point temperature error, K. Note that a fractional
        error in T becomes twice that fractional error in cooling power.
    tau : float
        Thermal time constant, s (from :func:`fit_approach`).

    Returns
    -------
    float
        Threshold ``s``, K/s.
    """
    return delta_t / tau


def choose_window(
    windows: ArrayLike,
    scatter: ArrayLike,
    s: float,
    tau: float,
    margin: float = 3.0,
    max_frac: float = 1 / 3,
) -> float | None:
    """Choose the slope-fitting window length.

    Picks the shortest window that satisfies both:

    * ``sigma_slope <= s / margin`` -- the slope estimate is precise
      enough that passing the threshold is a real detection, not noise;
    * ``W <= max_frac * tau`` -- the window is short enough that curvature
      of the exponential approach does not bias the slope.

    The shortest qualifying window is preferred because it responds
    fastest.

    Parameters
    ----------
    windows : array_like
        Candidate window lengths, s (as passed to :func:`slope_scatter`).
    scatter : array_like
        Slope standard deviation for each window, K/s, from
        :func:`slope_scatter`.
    s : float
        Slope threshold, K/s, from :func:`slope_threshold`.
    tau : float
        Thermal time constant, s.
    margin : float, optional
        Required ratio ``s / sigma_slope`` (default 3).
    max_frac : float, optional
        Maximum window as a fraction of ``tau`` (default 1/3).

    Returns
    -------
    float or None
        Chosen window, s, or ``None`` if no candidate satisfies both
        conditions. In that case, relax ``delta_t`` (raising ``s``) or
        accept a longer window and document the trade-off.
    """
    w, sc = _arr(windows), _arr(scatter)
    ok = (sc <= s / margin) & (w <= max_frac * tau)
    return float(w[ok].min()) if ok.any() else None


def is_settled(
    t: ArrayLike, temp: ArrayLike, s: float, window: float, hold: float
) -> tuple[bool, FloatArray]:
    """Decide whether the temperature has settled at the current setting.

    Fits straight-line slopes in windows of length ``window``, stepped
    back from the most recent sample in half-window increments until the
    windows span the last ``hold`` seconds, and requires every one to
    satisfy ``|dT/dt| < s``. Requiring the criterion to persist over
    ``hold`` (recommended: at least ``tau``) prevents a momentary flat
    spot, such as a noise excursion or the turning point of a slow
    oscillation, from triggering the next step.

    The windows are anchored to the latest sample, so the newest data is
    always tested. With ``hold = tau`` the earliest possible "settled"
    result is ``hold + window`` seconds after the step, but in practice it
    comes later: the oldest window must also pass, i.e. the slope ``hold``
    seconds ago must already be below ``s``.

    Parameters
    ----------
    t : array_like
        Time since the last heater change, s. Must start at (or after)
        the change, so that ``t[-1] - t[0]`` measures elapsed time.
    temp : array_like
        MC temperature, K. NaN samples (e.g. dropped or out-of-range
        readings) are ignored.
    s : float
        Slope threshold, K/s.
    window : float
        Slope-fitting window, s (from :func:`choose_window`).
    hold : float
        Time over which the criterion must hold, s.

    Returns
    -------
    settled : bool
        True if all windows pass. False if any fails, or if less than
        ``hold + window`` seconds have elapsed.
    slopes : ndarray
        Fitted slopes in chronological order (oldest first), K/s; about
        ``2 * hold / window + 1`` of them. Empty only while less than
        ``hold + window`` seconds of data exist. Windows with fewer than 3
        valid samples are skipped. Inspect the signs: consistently one
        sign means the approach is still under way even if magnitudes are
        small; signs scattered around zero indicate genuine settling.
    """
    t, temp = _arr(t), _arr(temp)
    ok = np.isfinite(t) & np.isfinite(temp)
    t, temp = t[ok], temp[ok]
    if len(t) < 3 or t[-1] - t[0] < hold + window:
        return False, np.array([])
    ends = t[-1] - np.arange(0.0, hold + 1e-9, window / 2)
    slopes = []
    for end in ends[::-1]:
        sel = (t > end - window) & (t <= end)
        if sel.sum() >= 3:
            ts = t[sel]
            slopes.append(np.polyfit(ts - ts.mean(), temp[sel], 1)[0])
    slopes = np.array(slopes)
    return bool(len(slopes) and np.all(np.abs(slopes) < s)), slopes


# ---------------------------------------------------------------------------
# 4. Planning
# ---------------------------------------------------------------------------


def estimate_a(power: float, temp: float, t_base: float) -> float:
    """Estimate the cooling coefficient ``a`` from one settled heater step.

    From ``P = a * T^2 - Q_par`` and ``Q_par = a * T_base^2`` (the base
    point), a single step gives ``a = P / (T^2 - T_base^2)``. Used to plan
    the heater powers for the full sweep.

    Parameters
    ----------
    power : float
        Applied MC heater power, W.
    temp : float
        Settled MC temperature at that power, K.
    t_base : float
        Base temperature with zero heater power, K.

    Returns
    -------
    float
        Cooling coefficient ``a``, W/K^2.
    """
    return power / (temp**2 - t_base**2)


def heater_plan(
    a: float, t_base: float, t_max: float, n_points: int = 10, n_repeat: int = 3
) -> tuple[FloatArray, FloatArray]:
    """Plan MC heater powers for the sweep, plus repeat points.

    Because ``P = a * (T^2 - T_base^2)``, powers evenly spaced in P are
    also evenly spaced in T^2, which is the variable the cooling-power fit
    is linear in. Evenly spaced temperatures would instead bunch points at
    the hot end in T^2.

    Repeat points revisit earlier powers on the way back down, ending at
    zero heater. Agreement between first and repeat visits confirms the
    background load did not drift during the sweep (see
    :func:`repeat_check`).

    Parameters
    ----------
    a : float
        Cooling coefficient, W/K^2 (from :func:`estimate_a`).
    t_base : float
        Base temperature, K.
    t_max : float
        Highest target temperature, K (e.g. 3 * t_base).
    n_points : int, optional
        Number of up-sweep points including 0 W (default 10).
    n_repeat : int, optional
        Number of repeat points on the way down (default 3). Set 0 to
        skip.

    Returns
    -------
    up : ndarray
        Up-sweep heater powers, W, from 0 to ``a * (t_max^2 - t_base^2)``.
    repeats : ndarray
        Descending repeat powers, W, ending at 0. Empty if
        ``n_repeat <= 0``. Duplicates are merged, so it may be shorter
        than ``n_repeat`` for small ``n_points``.
    """
    p_max = a * (t_max**2 - t_base**2)
    up = np.linspace(0.0, p_max, n_points)
    if n_repeat <= 0:
        return up, np.array([])
    idx = np.linspace(len(up) - 2, 0, n_repeat).round().astype(int)
    return up, up[np.unique(idx)[::-1]]


def predicted_temperature(power: ArrayLike, a: float, t_base: float) -> FloatArray:
    """Predict the settled MC temperature for given heater powers.

    Inverts ``P = a * (T^2 - T_base^2)``. Useful for sanity-checking a
    plan before running it and for spotting points that land unexpectedly
    far from prediction during the sweep.

    Parameters
    ----------
    power : array_like
        MC heater power, W.
    a : float
        Cooling coefficient, W/K^2.
    t_base : float
        Base temperature, K.

    Returns
    -------
    ndarray
        Predicted temperature ``sqrt(T_base^2 + P / a)``, K.
    """
    return np.sqrt(t_base**2 + _arr(power) / a)


# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------


def _linfit(x, y, sigma=None):
    """Weighted least-squares fit of ``y = m * x + b``.

    Parameters
    ----------
    x, y : array_like
        Data.
    sigma : array_like, optional
        1-sigma uncertainties of ``y``. If given, points are weighted by
        ``1 / sigma^2`` and the covariance uses these absolute errors. If
        omitted, all points are weighted equally and the covariance is
        scaled by the reduced chi-squared (same convention as
        ``scipy.optimize.curve_fit`` with ``absolute_sigma=False``).

    Returns
    -------
    m, b : float
        Slope and intercept.
    cov : ndarray, shape (2, 2)
        Covariance matrix of ``(m, b)``.
    chi2_red : float
        Reduced chi-squared. With absolute ``sigma``, values well above 1
        mean the errors are underestimated or the model misfits.
    """
    x, y = _arr(x), _arr(y)
    w = np.ones_like(y) if sigma is None else 1.0 / _arr(sigma) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    m, b = cov @ (A.T @ (w * y))
    res = y - (m * x + b)
    dof = max(len(y) - 2, 1)
    chi2_red = float(np.sum(w * res**2) / dof)
    if sigma is None:
        cov = cov * chi2_red
    return float(m), float(b), cov, chi2_red


@dataclass
class CoolingFit:
    """Result of fitting ``P_heater = a * T^2 - Q_par``.

    When fitted per unit flow (``per_flow=True``), the model is
    ``P_heater / ndot = a * T^2 - Q_par`` and the attributes are per mole.

    Attributes
    ----------
    a : float
        Cooling coefficient, W/K^2 (or J/(mol K^2) if ``per_flow``).
        Compare with the ideal ``84 * ndot`` via :func:`exchanger_ratio`.
    a_err : float
        1-sigma uncertainty of ``a``.
    Q_par : float
        Effective parasitic load on the MC, W (or J/mol if ``per_flow``;
        multiply by the mean flow for watts). Includes both true heat
        leaks and heat-exchanger inefficiency.
    Q_par_err : float
        1-sigma uncertainty of ``Q_par``.
    chi2_red : float
        Reduced chi-squared of the fit.
    T_base_pred : float
        Base temperature implied by the fit, ``sqrt(Q_par / a)``, K. NaN
        if the fit gives non-physical (non-positive) values. Compare with
        the measured base temperature as a consistency check.
    used : ndarray of bool
        Which input points were included in the fit.
    residuals : ndarray
        ``y - (a * T^2 - Q_par)`` for all input points, including those
        excluded from the fit. Systematic curvature (e.g. high-T points
        falling below the line) signals flow changes or the T_in term.
    per_flow : bool
        Whether the fit was normalized by flow.
    x, y : ndarray or None
        Fitted variables for all input points: ``T^2`` (K^2) and the
        heater power (W) or power per flow (J/mol). Stored for plotting.
    sigma : ndarray or None
        Per-point uncertainty of ``y`` used as fit weights, or ``None`` if
        the fit was unweighted.
    """

    a: float
    a_err: float
    Q_par: float
    Q_par_err: float
    chi2_red: float
    T_base_pred: float
    used: NDArray[np.bool_]
    residuals: FloatArray = field(repr=False)
    per_flow: bool = False
    x: FloatArray | None = field(default=None, repr=False)
    y: FloatArray | None = field(default=None, repr=False)
    sigma: FloatArray | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a one-line human-readable summary (mW/K^2, nW, mK)."""
        if self.per_flow:
            return (
                f"a' = {self.a:.4g} +/- {self.a_err:.2g} J/(mol K^2), "
                f"q_par = {self.Q_par:.4g} +/- {self.Q_par_err:.2g} J/mol, "
                f"chi2_red = {self.chi2_red:.2f}"
            )
        return (
            f"a = {self.a * 1e3:.4g} +/- {self.a_err * 1e3:.2g} mW/K^2, "
            f"Q_par = {self.Q_par * 1e9:.4g} +/- {self.Q_par_err * 1e9:.2g} nW, "
            f"T_base(pred) = {self.T_base_pred * 1e3:.2f} mK, "
            f"chi2_red = {self.chi2_red:.2f}"
        )


def fit_cooling_power(
    temp: ArrayLike,
    power: ArrayLike,
    sigma_t: ArrayLike | float | None = None,
    sigma_p: ArrayLike | float | None = None,
    flow: ArrayLike | None = None,
    t_max: float | None = None,
) -> CoolingFit:
    """Extract the MC parasitic load from settled heater-sweep points.

    Fits ``P_heater = a * T^2 - Q_par`` by weighted linear least squares
    in ``x = T^2``. The intercept at ``T^2 = 0`` is ``-Q_par``, the
    effective parasitic heat load on the mixing chamber.

    Parameters
    ----------
    temp : array_like
        Settled MC temperatures, K. Use ``T_inf`` from
        :func:`fit_approach` if extrapolating.
    power : array_like
        Applied MC heater powers, W. Include the 0 W base point.
    sigma_t : float or array_like, optional
        1-sigma temperature uncertainty per point, K. Propagated to the
        fitted variable as ``2 * a * T * sigma_T`` (one iteration, using
        an unweighted fit for the initial ``a``).
    sigma_p : float or array_like, optional
        1-sigma heater power uncertainty per point, W.
    flow : array_like, optional
        Circulation rate per point, mol/s. If given, fits
        ``P / ndot`` vs ``T^2`` so that flow changes across the sweep do
        not masquerade as extra load. Results are then per mole.
    t_max : float, optional
        Exclude points above this temperature, K. Use when high-T points
        curve below the line because the heat exchangers cannot keep up
        (T_in term) or the flow shifts.

    Returns
    -------
    CoolingFit
        Fitted ``a``, ``Q_par``, uncertainties, reduced chi-squared,
        predicted base temperature, and residuals.

    Notes
    -----
    If neither ``sigma_t`` nor ``sigma_p`` is given, points are weighted
    equally and uncertainties are scaled by the reduced chi-squared. If
    either is given, uncertainties are absolute; then ``chi2_red`` well
    above 1 means the stated errors are too small or the model does not
    fit (check the residuals).
    """
    temp, power = _arr(temp), _arr(power)
    used = np.ones_like(temp, dtype=bool) if t_max is None else temp <= t_max
    y = power if flow is None else power / _arr(flow)
    x = temp**2

    def sig(a_guess):
        """Combined per-point uncertainty in the fitted variable y."""
        if sigma_t is None and sigma_p is None:
            return None
        st = (
            np.zeros_like(temp)
            if sigma_t is None
            else np.broadcast_to(sigma_t, temp.shape)
        )
        sp = (
            np.zeros_like(temp)
            if sigma_p is None
            else np.broadcast_to(sigma_p, temp.shape)
        )
        if flow is not None:
            sp = sp / _arr(flow)
        return np.sqrt(sp**2 + (2 * a_guess * temp * st) ** 2)

    m, b, _, _ = _linfit(x[used], y[used])
    s = sig(m)
    if s is not None:
        m, b, cov, chi2 = _linfit(x[used], y[used], s[used])
    else:
        m, b, cov, chi2 = _linfit(x[used], y[used])
    q_par = -b
    t_base = np.sqrt(q_par / m) if q_par > 0 and m > 0 else np.nan
    res = y - (m * x + b)
    return CoolingFit(
        a=m,
        a_err=float(np.sqrt(cov[0, 0])),
        Q_par=q_par,
        Q_par_err=float(np.sqrt(cov[1, 1])),
        chi2_red=chi2,
        T_base_pred=float(t_base),
        used=used,
        residuals=res,
        per_flow=flow is not None,
        x=x,
        y=y,
        sigma=None if s is None else np.asarray(s, dtype=float),
    )


def exchanger_ratio(a: float, flow: float) -> float:
    """Compare the fitted cooling coefficient with the ideal value.

    For perfect heat exchangers, the cooling coefficient is
    ``84 * ndot_3`` (W/K^2 with ndot in mol/s). The ratio of the measured
    ``a`` to this ideal value indicates how well the circulation is being
    used.

    Parameters
    ----------
    a : float
        Fitted cooling coefficient, W/K^2 (from :func:`fit_cooling_power`
        without ``flow``).
    flow : float
        Measured circulation rate, mol/s (mean over the sweep).

    Returns
    -------
    float
        ``a / (84 * flow)``. Near 1: near-ideal exchangers. Well below 1:
        poor exchanger performance, or a substantial 4He fraction in the
        measured flow (which then only bounds the 3He circulation from
        above).
    """
    return a / (IDEAL_COEFF * flow)


def repeat_check(
    temp_first: ArrayLike, temp_repeat: ArrayLike, sigma: ArrayLike | float
) -> FloatArray:
    """Test whether repeat points agree with their first visit.

    Revisiting heater powers at the end of the sweep checks for drift in
    the background heat load (slow heat release, cryocooler changes) and
    for hysteresis. Disagreement means the extrapolated ``Q_par`` mixes
    different background conditions and should be treated with caution.

    Parameters
    ----------
    temp_first : array_like
        Settled temperatures from the first visit, K.
    temp_repeat : array_like
        Settled temperatures from the repeat visit at the same powers, K.
    sigma : float or array_like
        1-sigma uncertainty of a single settled temperature, K (assumed
        equal for both visits).

    Returns
    -------
    ndarray
        z-scores ``(T_repeat - T_first) / (sqrt(2) * sigma)``. Values of
        ``|z|`` well above ~2 indicate a real change. A consistent sign
        across points suggests a steadily drifting background.
    """
    d = _arr(temp_repeat) - _arr(temp_first)
    return d / (np.sqrt(2.0) * _arr(sigma))


@dataclass
class StillFit:
    """Result of fitting ``ndot = (P_still + Q_par) / L_eff``.

    Attributes
    ----------
    L_eff : float
        Effective latent heat per mole circulated, J/mol. Roughly 40 J/mol
        is expected for pure 3He evaporation near 0.7 K; lower values
        suggest significant 4He in the circulating gas.
    L_eff_err : float
        1-sigma uncertainty of ``L_eff``, J/mol.
    Q_par : float
        Parasitic heat load on the still, W (zero-flow intercept).
    Q_par_err : float
        1-sigma uncertainty of ``Q_par``, W, including the slope-intercept
        correlation.
    chi2_red : float
        Reduced chi-squared of the fit.
    power, flow : ndarray or None
        Fitted still heater powers (W) and flows (mol/s). Stored for
        plotting.
    sigma : ndarray or None
        Flow uncertainties used as weights, or ``None`` if unweighted.
    """

    L_eff: float
    L_eff_err: float
    Q_par: float
    Q_par_err: float
    chi2_red: float
    power: FloatArray | None = field(default=None, repr=False)
    flow: FloatArray | None = field(default=None, repr=False)
    sigma: FloatArray | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a one-line human-readable summary (J/mol and uW)."""
        return (
            f"L_eff = {self.L_eff:.3g} +/- {self.L_eff_err:.2g} J/mol, "
            f"Q_par(still) = {self.Q_par * 1e6:.4g} +/- "
            f"{self.Q_par_err * 1e6:.2g} uW, chi2_red = {self.chi2_red:.2f}"
        )


def fit_still_load(
    still_power: ArrayLike,
    flow: ArrayLike,
    sigma_flow: ArrayLike | float | None = None,
) -> StillFit:
    """Extrapolate flow vs. still heater power to estimate the still load.

    Circulation scales with the total heat into the still,
    ``ndot = (P_still + Q_par) / L_eff``. A linear fit of flow against
    applied still heater power gives ``1 / L_eff`` as the slope, and
    extrapolating to zero flow gives ``-Q_par`` on the power axis. This
    works without any historical still-heater data.

    Parameters
    ----------
    still_power : array_like
        Applied still heater powers, W (include 0 W).
    flow : array_like
        Settled circulation rate at each power, mol/s.
    sigma_flow : float or array_like, optional
        1-sigma flow uncertainty, mol/s. If omitted, points are weighted
        equally and uncertainties are scaled by the reduced chi-squared.

    Returns
    -------
    StillFit
        ``L_eff``, still parasitic load ``Q_par``, uncertainties, and
        reduced chi-squared.

    Notes
    -----
    If 4He is circulating, the measured flow overstates the 3He flow and
    ``L_eff`` comes out low; ``Q_par`` is then approximate. Keep the still
    below ~0.85 K and watch the condensing pressure while sweeping.
    """
    p, n = _arr(still_power), _arr(flow)
    sig = None if sigma_flow is None else np.broadcast_to(sigma_flow, n.shape)
    m, b, cov, chi2 = _linfit(p, n, sig)
    L = 1.0 / m
    L_err = np.sqrt(cov[0, 0]) / m**2
    q = b / m
    # Q_par = b / m; propagate errors including the slope/intercept covariance.
    jac = np.array([-b / m**2, 1.0 / m])
    q_err = float(np.sqrt(jac @ cov @ jac))
    return StillFit(
        L,
        float(L_err),
        q,
        q_err,
        chi2,
        power=p,
        flow=n,
        sigma=None if sig is None else np.asarray(sig, dtype=float),
    )


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------


def _annotate(ax, lines: list[str], loc: str = "upper left") -> None:
    """Draw a boxed multi-line text annotation in an axes corner.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lines : list of str
        Text lines (matplotlib mathtext allowed).
    loc : {"upper left", "upper right", "lower left", "lower right"}
        Corner to anchor the box in.
    """
    x = 0.03 if "left" in loc else 0.97
    y = 0.97 if "upper" in loc else 0.03
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _fmt_power(value: float, err: float) -> str:
    """Format a power and its uncertainty with an auto-chosen SI prefix.

    Picks nW, µW, or mW from the magnitude so annotations stay readable
    (e.g. ``36.02 ± 0.05 µW`` rather than ``3.602e+04 nW``).
    """
    mag = max(abs(value), abs(err))
    for scale, unit in ((1e9, "nW"), (1e6, "µW"), (1e3, "mW")):
        if mag * scale < 1e3:
            return f"{value * scale:.4g} ± {err * scale:.2g} {unit}"
    return f"{value:.4g} ± {err:.2g} W"


def _fit_quality(chi2_red: float, weighted: bool) -> str:
    """Annotation line for fit quality.

    Reduced chi-squared is only meaningful for a weighted fit; for an
    unweighted fit it is just the mean squared residual, so say so instead.
    """
    if weighted:
        return f"$\\chi^2_\\nu$ = {chi2_red:.2f}"
    return "unweighted fit (no $\\chi^2$)"


def plot_cooling_fit(
    fit: CoolingFit,
    repeats: tuple[ArrayLike, ArrayLike] | None = None,
    title: str | None = None,
):
    """Plot an MC cooling-power fit with residuals and annotations.

    Top panel: heater power vs. ``T^2`` with error bars, the fitted line
    extended to ``T^2 = 0`` (dashed) to show the extrapolation, the
    intercept ``-Q_par`` marked, and the predicted base temperature
    marked on the zero-power line. A secondary top axis shows ``T`` in mK.
    Points excluded by ``t_max`` are drawn hollow. Bottom panel: residuals,
    where systematic curvature reveals flow changes or the T_in term.

    Parameters
    ----------
    fit : CoolingFit
        Result of :func:`fit_cooling_power` (must carry ``x``/``y``, as
        all fits from this version do).
    repeats : tuple of (temp, power), optional
        Repeat points (K, W) to overlay as open diamonds, for checking
        drift by eye alongside :func:`repeat_check`. Ignored for per-flow
        fits.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    (ax, ax_res) : tuple of matplotlib.axes.Axes
        Main and residual axes, for further customization.
    """
    import matplotlib.pyplot as plt

    if fit.x is None or fit.y is None:
        raise ValueError("fit has no stored data; refit with fit_cooling_power")
    xs = fit.x * 1e6  # K^2 -> mK^2
    yscale, yunit = (1.0, "J/mol") if fit.per_flow else (1e6, "µW")
    ys = fit.y * yscale
    es = None if fit.sigma is None else fit.sigma * yscale
    res = fit.residuals * yscale
    used = fit.used

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    for mask, face, label in ((used, "C0", "fitted"), (~used, "none", "excluded")):
        if mask.any():
            ax.errorbar(
                xs[mask],
                ys[mask],
                yerr=None if es is None else es[mask],
                fmt="o",
                mfc=face,
                mec="C0",
                ecolor="C0",
                capsize=3,
                label=label,
            )
            axr.errorbar(
                xs[mask],
                res[mask],
                yerr=None if es is None else es[mask],
                fmt="o",
                mfc=face,
                mec="C0",
                ecolor="C0",
                capsize=3,
            )
    if repeats is not None and not fit.per_flow:
        tr, pr = _arr(repeats[0]), _arr(repeats[1])
        ax.plot(tr**2 * 1e6, pr * 1e6, "D", mfc="none", mec="C3", label="repeats")

    xline = np.linspace(0.0, xs.max() * 1.05, 200)
    yline = (fit.a * xline * 1e-6 - fit.Q_par) * yscale
    ax.plot(xline, yline, "C1--", lw=1.2, label="fit (extrapolated)")
    xfit = np.linspace(xs[used].min(), xs[used].max(), 100)
    ax.plot(xfit, (fit.a * xfit * 1e-6 - fit.Q_par) * yscale, "C1-", lw=2)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.plot(0, -fit.Q_par * yscale, "s", color="C3", ms=8, label="$-Q_{par}$")
    if np.isfinite(fit.T_base_pred):
        ax.plot(
            fit.T_base_pred**2 * 1e6,
            0,
            "v",
            color="C2",
            ms=9,
            label="$T_{base}$ (pred.)",
        )

    if fit.per_flow:
        lines = [
            f"$a'$ = {fit.a:.4g} ± {fit.a_err:.2g} J/(mol K²)",
            f"$q_{{par}}$ = {fit.Q_par:.4g} ± {fit.Q_par_err:.2g} J/mol",
            _fit_quality(fit.chi2_red, fit.sigma is not None),
        ]
        ax.set_ylabel("$P_{heater}/\\dot n$ (J/mol)")
    else:
        lines = [
            f"$Q_{{par}}$ = {_fmt_power(fit.Q_par, fit.Q_par_err)}",
            f"$a$ = {fit.a * 1e3:.4g} ± {fit.a_err * 1e3:.2g} mW/K²",
            f"$T_{{base}}$ (pred.) = {fit.T_base_pred * 1e3:.2f} mK",
            _fit_quality(fit.chi2_red, fit.sigma is not None),
        ]
        ax.set_ylabel("$P_{heater}$ (µW)")
    _annotate(ax, lines)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    top = ax.secondary_xaxis(
        "top",
        functions=(lambda x: np.sqrt(np.clip(x, 0, None)), lambda t: t**2),
    )
    from matplotlib.ticker import MaxNLocator

    t_lo, t_hi = np.sqrt(xs.min()), np.sqrt(xline.max())
    ticks = MaxNLocator(6).tick_values(t_lo, t_hi)
    top.set_ticks(ticks[(ticks >= t_lo * 0.9) & (ticks <= t_hi)])
    top.set_xlabel("$T_{MC}$ (mK)")

    axr.axhline(0, color="C1", lw=1)
    axr.set_xlabel("$T_{MC}^2$ (mK²)")
    axr.set_ylabel(f"resid. ({yunit})")
    axr.grid(alpha=0.3)
    ax.set_xlim(left=-0.02 * xs.max())
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax, axr)


def plot_still_fit(fit: StillFit, title: str | None = None):
    """Plot a still-load fit: flow vs. still heater power, extrapolated.

    Shows the data with error bars, the fitted line extended down to zero
    flow (dashed), and the zero-flow intercept at ``P = -Q_par`` marked on
    the power axis. Annotated with ``L_eff`` (compared with ~40 J/mol for
    pure 3He), ``Q_par``, and reduced chi-squared.

    Parameters
    ----------
    fit : StillFit
        Result of :func:`fit_still_load`.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if fit.power is None or fit.flow is None:
        raise ValueError("fit has no stored data; refit with fit_still_load")
    p, n = fit.power * 1e6, fit.flow * 1e6  # uW, umol/s
    e = None if fit.sigma is None else fit.sigma * 1e6

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(p, n, yerr=e, fmt="o", capsize=3, label="data")
    p0 = -fit.Q_par * 1e6
    pl = np.linspace(min(p0, p.min()), p.max() * 1.05, 200)
    ax.plot(
        pl,
        (pl * 1e-6 + fit.Q_par) / fit.L_eff * 1e6,
        "C1--",
        label="fit (extrapolated)",
    )
    ax.plot(p0, 0, "s", color="C3", ms=8, label="$-Q_{par}$")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    lines = [
        f"$Q_{{par}}$(still) = {_fmt_power(fit.Q_par, fit.Q_par_err)}",
        f"$L_{{eff}}$ = {fit.L_eff:.3g} ± {fit.L_eff_err:.2g} J/mol (³He ≈ 40)",
        _fit_quality(fit.chi2_red, fit.sigma is not None),
    ]
    _annotate(ax, lines)
    ax.set_xlabel("$P_{still}$ (µW)")
    ax.set_ylabel("flow (µmol/s)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_allan(
    tau_avg: ArrayLike,
    adev: ArrayLike,
    fits: list[AllanSlope] | AllanSlope | None = None,
    units: tuple[float, str] = (1e6, "µK"),
    guides: bool = True,
    title: str | None = None,
):
    """Plot an Allan deviation with fitted slopes and noise-type guides.

    Parameters
    ----------
    tau_avg, adev : array_like
        Output of :func:`allan_deviation`.
    fits : AllanSlope or list of AllanSlope, optional
        Fitted regions from :func:`allan_slope`. Each is drawn over its
        range, shaded, and labeled with its slope and closest noise type.
    units : (scale, label), optional
        Display scaling for ``adev`` (default K -> µK).
    guides : bool, optional
        Draw faint reference lines with slopes -1/2, 0, +1/2, +1 through
        the curve's midpoint, to compare against by eye.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    tau, ad = _arr(tau_avg), _arr(adev) * units[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(tau, ad, "o-", ms=3, lw=1, label="Allan deviation")
    if guides:
        k = len(tau) // 2
        tg = np.array([tau.min(), tau.max()])
        for slope in ALLAN_NOISE_TYPES:
            ax.loglog(tg, ad[k] * (tg / tau[k]) ** slope, ":", color="0.6", lw=0.8)
            ax.annotate(
                f"{slope:+g}",
                (tg[-1], ad[k] * (tg[-1] / tau[k]) ** slope),
                fontsize=7,
                color="0.5",
            )
    if fits is not None:
        for i, f in enumerate([fits] if isinstance(fits, AllanSlope) else fits):
            c = f"C{i + 1}"
            tf = np.geomspace(*f.tau_range, 50)
            ax.loglog(
                tf,
                10**f.intercept * tf**f.slope * units[0],
                color=c,
                lw=2.5,
                label=f"{f.slope:+.2f} ± {f.slope_err:.2f} ({f.noise_type})",
            )
            ax.axvspan(*f.tau_range, color=c, alpha=0.08)
    ax.set_xlabel(r"averaging time $\tau$ (s)")
    ax.set_ylabel(f"Allan deviation ({units[1]})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    lo, hi = ad[np.isfinite(ad) & (ad > 0)].min(), ad[np.isfinite(ad)].max()
    ax.set_ylim(lo / 3, hi * 3)
    if guides:
        # Label each guide where it leaves the plot (right edge or top/bottom).
        for slope in ALLAN_NOISE_TYPES:
            tx = tau.max()
            yx = ad[k] * (tx / tau[k]) ** slope
            if yx > hi * 3:
                tx = tau[k] * (hi * 3 / ad[k]) ** (1 / slope)
                yx = hi * 2.4
            elif yx < lo / 3:
                tx = tau[k] * (lo / 3 / ad[k]) ** (1 / slope)
                yx = lo / 2.6
            ax.annotate(f"{slope:+g}", (tx, yx), fontsize=7, color="0.45")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_approach(
    t: ArrayLike,
    temp: ArrayLike,
    fit: SettleFit,
    skip: float = 0.0,
    delta_t: float | None = None,
    title: str | None = None,
):
    """Plot a heater-step approach to steady state with its exponential fit.

    Shows the data, the fitted ``T_inf + A exp(-t/tau)`` curve, the
    extrapolated ``T_inf`` with its uncertainty band, the skipped initial
    transient (shaded), and markers at 1, 3 and 5 tau. If ``delta_t`` is
    given, the band ``T_inf +/- delta_t`` shows the accuracy target, and
    the annotation includes the predicted settling time
    ``tau * ln(|A| / delta_t)`` and the threshold ``s = delta_t / tau``.

    Parameters
    ----------
    t : array_like
        Time, s, as passed to :func:`fit_approach`.
    temp : array_like
        MC temperature, K.
    fit : SettleFit
        Result of :func:`fit_approach` on the same data.
    skip : float, optional
        The ``skip`` used in the fit, s (shaded on the plot).
    delta_t : float, optional
        Per-point accuracy target, K.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    t, temp = _arr(t), _arr(temp)
    tr = t - t[0]
    t_fit0 = tr[tr >= skip][0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tr / 60, temp * 1e3, ".", ms=3, color="C0", label="data")
    tm = np.linspace(t_fit0, tr[-1], 300)
    model = fit.T_inf + fit.amplitude * np.exp(-(tm - t_fit0) / fit.tau)
    ax.plot(tm / 60, model * 1e3, "C1-", lw=2, label="exponential fit")
    ax.axhline(fit.T_inf * 1e3, color="C2", lw=1, label="$T_\\infty$")
    ax.axhspan(
        (fit.T_inf - fit.T_inf_err) * 1e3,
        (fit.T_inf + fit.T_inf_err) * 1e3,
        color="C2",
        alpha=0.25,
    )
    if delta_t is not None:
        ax.axhspan(
            (fit.T_inf - delta_t) * 1e3,
            (fit.T_inf + delta_t) * 1e3,
            color="C2",
            alpha=0.08,
            label="± target",
        )
    if skip > 0:
        ax.axvspan(0, skip / 60, color="0.5", alpha=0.15, label="skipped")
    for k in (1, 3, 5):
        tk = t_fit0 + k * fit.tau
        if tk <= tr[-1] * 1.2:
            ax.axvline(tk / 60, color="0.6", ls=":", lw=0.8)
            ax.annotate(
                f"{k}τ",
                (tk / 60, 0.02),
                xycoords=("data", "axes fraction"),
                fontsize=8,
                color="0.4",
            )
    lines = [
        f"$T_\\infty$ = {fit.T_inf * 1e3:.3f} ± {fit.T_inf_err * 1e3:.3f} mK",
        f"τ = {fit.tau:.0f} ± {fit.tau_err:.0f} s ({fit.tau / 60:.1f} min)",
    ]
    if delta_t is not None:
        t_set = fit.tau * np.log(max(abs(fit.amplitude) / delta_t, 1.0))
        lines += [
            f"settle to ±{delta_t * 1e3:.2g} mK: {t_set / 60:.0f} min after fit start",
            f"threshold s = {delta_t / fit.tau * 1e6:.2g} µK/s",
        ]
    _annotate(ax, lines, loc="lower right" if fit.amplitude < 0 else "upper right")
    ax.set_xlabel("time since step (min)")
    ax.set_ylabel("$T_{MC}$ (mK)")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax
