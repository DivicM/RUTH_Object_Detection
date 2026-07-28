from car.config import SERIAL_BAUDRATE, SERIAL_PORT

try:
    import serial
except ImportError:
    serial = None


class MotorController:
    """Sends steering commands to an Arduino over serial.

    Falls back to a no-op if pyserial isn't installed or the port can't be
    opened, so the vision/FSM pipeline can still run and be tested without
    hardware attached.
    """

    def __init__(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
        self._last_command = None
        self._link = None

        if serial is None:
            return

        try:
            self._link = serial.Serial(port, baudrate, timeout=1)
        except serial.SerialException:
            self._link = None

    @property
    def connected(self):
        return self._link is not None

    def send(self, command):
        if command == self._last_command:
            return

        self._last_command = command

        if self._link is not None:
            self._link.write(f"{command}\n".encode("utf-8"))

    def close(self):
        if self._link is not None:
            self._link.close()
