"""
Uses the Lakeshore 218 temperature controller as an Ohmmeter.
"""
import serial
import time
from datetime import datetime as dt, timezone
from pathlib import Path
import sys

from serial.serialutil import SerialTimeoutException

LS218_BAUD = 9600
LS218_STOP_BITS = serial.STOPBITS_ONE
LS218_PARITY = serial.PARITY_ODD
LS218_CHARBITS = serial.SEVENBITS
LS218_COM = "/dev/ttyUSB0"
CSV_PATH = Path.cwd() / f"resistance_data_{dt.now(timezone.utc).replace(microsecond=0).isoformat()}"
LS218_TERM = '\r\n'
READ_TIMEOUT = 2.0
CSV_HEADER = "timestamp,coil_res,coax_res,stage_res\n"
COIL_CHANNEL = 1
COAX_CHANNEL = 5
STAGE_CHANNEL = 2
CSV_WRITE_THRESHOLD = 100
POLL_INTERVAL = 1

def _write(handle: serial.Serial, msg: str) -> int:
    """
    Low level internal function that performs transmission
    of bytes after formatting.
    """
    if (ret := handle.write(f'{msg}{LS218_TERM}'.encode('ascii'))):
        return ret
    else:
        return 0


def _query(handle: serial.Serial, scpi_msg: str)-> str:
    """
    Sends a SCPI query to the device and returns the
    response.
    """
    _ = _write(handle, scpi_msg)


    # Poll local read buffer until terminator is read
    rx_buf = handle.read_until(LS218_TERM.encode('ascii')) # Will block indefinitely unless timeout is set!
    rx_msg = rx_buf.decode('ascii').rstrip() # strip terminator
    return rx_msg

def poll_sensor_units(handle: serial.Serial) -> tuple[float] | None:
    """
    Polls the device for the sensor unit reading for each channel (1-8). The index of the sensor reading value in the
    returned tuple maps to channel-1. E.g. index 0 -> channel 1.

    If timeout or malformed message (missing sensor value), returns None
    """
    # We expect to see 8 values in the return message. If we do not, then the device
    # did not respond before the timeout, throw this data away
    vals = _query(handle, "SRDG? 0")
    vals = vals.split(",")

    # We expect to see 8 values in the return message. If we do not, then the device
    # did not respond before the timeout, throw this data away
    if len(vals) != 8:
        return None
    else:
        return tuple([float(val) for val in vals])

def format_csv(timestamp: str, readings: list[float]) -> str:
    ret = timestamp + ","
    for m in readings:
        ret += str(m) + ","
    ret = ret.rstrip(",")
    return ret + "\n"

def writelines_csv(file_path, readings: list[str]):
    with open(file_path,"a+") as f:
        f.writelines(readings)

def main():
    # Create serial handle and connect to device
    ls218_handle = serial.Serial(
        LS218_COM, LS218_BAUD, LS218_CHARBITS,
        LS218_PARITY, LS218_STOP_BITS, timeout=READ_TIMEOUT)
    good_reads = 0
    total_reads = 0
    readings = []
    with open(CSV_PATH, "w") as f:
        f.write(CSV_HEADER)

    try:
        while True:
            try:
                if (ret := poll_sensor_units(ls218_handle)):
                    ts = dt.now(timezone.utc).replace(microsecond=0).isoformat()
                    print(f"Coax: {ret[COAX_CHANNEL-1]} Ohms, Coil: {ret[COIL_CHANNEL-1]} Ohms, Stage: {ret[STAGE_CHANNEL-1]} Ohms")
                    readings.append(format_csv(ts, [ret[COIL_CHANNEL-1], ret[COAX_CHANNEL-1], ret[STAGE_CHANNEL-1]]))

                    if len(readings) >= CSV_WRITE_THRESHOLD:
                        writelines_csv(CSV_PATH, readings)
                        readings.clear()
                    good_reads += 1

                else:
                    print(ret)

                total_reads += 1
                print(f"Good reads ratio: {good_reads}/{total_reads}\n")
                time.sleep(POLL_INTERVAL)

            except SerialTimeoutException:
                continue

    except KeyboardInterrupt:
        print("Got keyboard interrupt, cleaning up...")
        sys.exit(0)

    finally:
        writelines_csv(CSV_PATH, readings)


if __name__ == "__main__":
    main()
