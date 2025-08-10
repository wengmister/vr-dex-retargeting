from xhand_controller import xhand_control
import time
import math
import sys

class XHandControlExample:
    def __init__(self, hand_id=0, position=0.1, mode=3):
        self._hand_id = hand_id
        self._device = xhand_control.XHandControl()
        self._hand_command = xhand_control.HandCommand_t()
        for i in range(12):
            self._hand_command.finger_command[i].id = i
            self._hand_command.finger_command[i].kp = 100
            self._hand_command.finger_command[i].ki = 0
            self._hand_command.finger_command[i].kd = 0
            self._hand_command.finger_command[i].position = position
            self._hand_command.finger_command[i].tor_max = 300
            self._hand_command.finger_command[i].mode = mode

    def exam_enumerate_devices(self, protocol: str):
        print("//================================")
        print("//Enumerate device hardware input ports")
        print("//================================")
        serial_port = self._device.enumerate_devices(protocol)
        print(f"=@= xhand devices port: {serial_port}\n")
        return serial_port

    def exam_open_device(self, device_identifier: dict):
        print("//================================")
        print("//Open hand device")
        print("//================================")
        # RS485
        if device_identifier["protocol"] == "RS485":
            device_identifier["baud_rate"] = int(device_identifier["baud_rate"])
            rsp = self._device.open_serial(
                device_identifier["serial_port"],
                device_identifier["baud_rate"],
            )
            print(f"=@= open RS485 result: {rsp.error_code == 0}\n")
        # EtherCAT
        elif device_identifier["protocol"] == "EtherCAT":
            ether_cat = self.exam_enumerate_devices("EtherCAT")
            print(f"enumerate_devices_ethercat ether_cat= {ether_cat}\n")
            if ether_cat is None or not ether_cat:
                print("enumerate_devices_ethercat get empty \n")
            rsp = self._device.open_ethercat(ether_cat[0])
            print(f"=@= open EtherCAT result: {rsp.error_code == 0}\n")
        if rsp.error_code != 0:
            print(f"=@= open device error: {rsp.error_message}. Please check serial_port and connection\n")
            return False
        else:
            return True

    def exam_list_hands_id(self):
        print("//================================")
        print("//List hand IDs")
        print("//================================")
        self._hand_id = self._device.list_hands_id()[0]
        print(f"=@= hand_id: {self._hand_id}\n")

    def exam_set_hand_id(self, new_id):
        print("//================================")
        print("//Set hand ID")
        print("//================================")
        print(f"set hand_id from {self._hand_id} to {new_id}\n")
        hands_id = self._device.list_hands_id()
        old_id = hands_id[0]
        err_struct = self._device.set_hand_id(old_id, new_id)
        if err_struct.error_code == 0:
            self._hand_id = new_id
        hands_id = self._device.list_hands_id()
        print(f"=@= xhand set_hand_id result: {err_struct.error_code == 0}\n")

    def exam_set_hand_name(self, new_name):
        print("//================================")
        print("//Set hand name")
        print("//================================")
        error_struct, old_name = self._device.get_hand_name(self._hand_id)
        print(f"set hand_name from '{old_name}' to '{new_name}'\n")
        err_struct = self._device.set_hand_name(self._hand_id, new_name)
        print(f"=@= xhand set_hand_name result: {err_struct.error_code == 0}\n")

    def exam_get_hand_name(self):
        print("//================================")
        print("//Get hand name")
        print("//================================")
        error_struct, hand_name = self._device.get_hand_name(self._hand_id)
        print(f"=@= xhand hand_name: '{hand_name}', error_code: {error_struct.error_code}\n")

    def exam_read_device_info(self):
        print("//================================")
        print("//Read hand device information")
        print("//================================")
        error_struct, info = self._device.read_device_info(self._hand_id)
        print(f"=@= xhand serial_number: {info.serial_number[0:16]}") # sn is 16 bytes
        print(f"=@= xhand hand_id: {info.hand_id}")
        print(f"=@= xhand hand_id: {info.ev_hand}\n")

    def exam_serial_number(self):
        print("//================================")
        print("//Read hand serial number")
        print("//================================")
        error_struct, serial_number = self._device.get_serial_number(self._hand_id)
        print(f"=@= xhand serial_number: {serial_number}\n")

    def exam_get_hand_type(self):
        print("//================================")
        print("//Get hand left/right type")
        print("//================================")
        error_struct, hand_type = self._device.get_hand_type(self._hand_id)
        print(f"=@= xhand hand_type: {hand_type}\n")

    def exam_read_version(self):
        print("//================================")
        print("//Read hardware SDK version")
        print("//================================")
        joint_id = 0
        error_struct, version = self._device.read_version(self._hand_id, joint_id)
        print(f"=@= xhand hardware SDK version: {version}\n")
            
    def exam_send_command(self):
        print("//================================")
        print("//Send hand activity command")
        print("//================================")
        error_struct = self._device.send_command(self._hand_id, self._hand_command)
        print(f"=@= xhand send_command result: { error_struct.error_code == 0}, message: {error_struct.error_message}\n")
        time.sleep(1)

    def exam_read_state(self, fingure_id=2, force_update=True):
        print("//================================")
        print("//Read various hand states")
        print("//================================")
        error_struct, state = self._device.read_state(self._hand_id, force_update)
        if error_struct.error_code != 0:
            print(f"=@= xhand read_state error: {error_struct.error_message}\n")
            return
        
        finger_1 = state.finger_state[fingure_id]
        print(f"|+| finger.id = {finger_1.id}, finger.temperature = {finger_1.temperature} ")
        print(f"|+| finger.id = {finger_1.id}, finger.temperature & 0xFF = {finger_1.temperature & 0xFF} ")
        print(f"|+| finger.id = {finger_1.id}, finger.commboard_err = {finger_1.commboard_err} ")
        print(f"|+| finger.id = {finger_1.id}, finger.jonitboard_err = {finger_1.jonitboard_err} ")
        print(f"|+| finger.id = {finger_1.id}, finger.tipboard_err = {finger_1.tipboard_err} ")

        # Fingertip sensor state
        fingertip_state = {}
        if finger_1.id in {2, 5, 7, 9, 11}:
            sensor_data = state.sensor_data[0]
            fingertip_state["calc_pressure"] = [
                sensor_data.calc_force.fx,
                sensor_data.calc_force.fy,
                sensor_data.calc_force.fz,
            ]
            fingertip_state["raw_pressure"] = [
                [force.fx, force.fy, force.fz]
                for force in state.sensor_data[0].raw_force
            ]
            fingertip_state["sensor_temperature"] = sensor_data.calc_temperature

        print(f"|+| finger.id = {finger_1.id}, fingertip calc_pressure = {fingertip_state['calc_pressure']}")
        print(f"|+| finger.id = {finger_1.id}, fingertip raw_pressure = {fingertip_state['raw_pressure']}")
        print(f"|+| finger.id = {finger_1.id}, fingertip sensor_temperature = {fingertip_state['sensor_temperature']}\n")
        print(f"=@= xhand read state result: {error_struct.error_code == 0} | error_struct.error_code={error_struct.error_code} error_msg={error_struct.error_message}\n")
    
    def exam_reset_sensor(self):
        print("//================================")
        print("//Reset fingertip sensor")
        print("//================================")
        sensor_id = 17
        print(f"=@= xhand reset_sensor result: {self._device.reset_sensor(self._hand_id, sensor_id).error_code == 0}\n")

    def exam_close_device(self):
        print("//================================")
        print("//Close hand device")
        print("//================================")
        print(f"=@= xhand device closed\n")
	
    def set_hand_mode(self, mode: int):
        print("//================================")
        print("//Set hand activity mode")
        print("//================================")
        hand_mode = xhand_control.HandCommand_t()
        for i in range(12):
            hand_mode.finger_command[i].id = i
            hand_mode.finger_command[i].kp = 120
            hand_mode.finger_command[i].ki = 0
            hand_mode.finger_command[i].kd = 0
            hand_mode.finger_command[i].position = 0.5
            hand_mode.finger_command[i].tor_max = 380
            hand_mode.finger_command[i].mode = mode
        error_struct = self._device.send_command(self._hand_id, hand_mode)
        print(f"=@= xhand set hand mode result: {error_struct.error_code == 0} | error_struct.error_code={error_struct.error_code} error_msg={error_struct.error_message}\n")
        time.sleep(1)

    def exam_get_sdk_version(self):
        print("//================================")
        print("//Read software SDK version")
        print("//================================")
        print(f"=@= xhand software SDK version: {self._device.get_sdk_version()}\n")


if __name__ == "__main__":
    # Default hand ID is 0, finger position is 0.1, mode is 3
    xhand_exam = XHandControlExample(hand_id=0, position=0.1, mode=3)

    # Choose one communication method, currently supports EtherCAT and RS485
    # First of all, open device 
    device_identifier = {}

    while True:
        communication_choice = input("Choose communication method (enter '1' for EtherCAT, enter '2' for RS485): ").strip()
        if communication_choice == '1':
            device_identifier['protocol'] = 'EtherCAT'
            if xhand_exam.exam_open_device(device_identifier):
                break
            else:
                sys.exit(1)
        elif communication_choice == '2':
            device_identifier['protocol'] = 'RS485'
            # You can use exam_enumerate_devices('RS485') to read serial port list information, choose ttyUSB prefixed port
            # Get serial port list, choose ttyUSB*
            xhand_exam.exam_enumerate_devices('RS485')
            device_identifier["serial_port"] = '/dev/ttyUSB0'
            device_identifier['baud_rate'] = 3000000
            if xhand_exam.exam_open_device(device_identifier):
                break
            else:
                sys.exit(1)
        else:
            print("Invalid choice, please enter '1' or '2'\n")

    # List hand IDs
    # xhand_exam.exam_list_hands_id()
    # Read software SDK version
    xhand_exam.exam_get_sdk_version()
    # Read hardware SDK version
    xhand_exam.exam_read_version()
    # Read hand device information
    xhand_exam.exam_read_device_info()
    # Get hand left/right type
    xhand_exam.exam_get_hand_type()
    # Read hand serial number
    xhand_exam.exam_serial_number()
    # Read various hand states
    # {2, 5, 7, 9, 11} are fingertip sensors
    # if not use send_command func, let force_update value to True to force update state
    xhand_exam.exam_read_state(fingure_id=5, force_update=True)

    # Reset fingertip sensor
    # xhand_exam.exam_reset_sensor()
    # Set hand ID
    # xhand_exam.exam_set_hand_id(new_id=0)

    # Set hand name
    # xhand_exam.exam_set_hand_name(new_name="xhand")
    # Get hand name
    # xhand_exam.exam_get_hand_name()

    # ================================================================================
    # !! Warning: This function will send motion control command to device ！！
    # ================================================================================
    # Set hand mode（0: powerless, 3: position (default), 5: powerful）
    # xhand_exam.set_hand_mode(mode=3)

    # ================================================================================
    # !! Warning: This function will send motion control command to device ！！
    # ================================================================================
    # Send hand control command
    # xhand_exam.exam_send_command()

    # ================================================================================
    # !! Warning: This function will send motion control command to device ！！
    # ================================================================================
    # Send xhand preset action list  
    actions_list = {
        'fist': [11.85, 74.58, 40, -3.08, 106.02, 110, 109.75, 107.56, 107.66, 110, 109.1, 109.15],
        'palm': [0, 80.66, 33.2, 0.00, 5.11, 0, 6.53, 0, 6.76, 4.41, 10.13, 0],
        'v': [38.32, 90, 52.08, 6.21, 2.6, 0, 2.1, 0, 110, 110, 110, 109.23],
        'ok': [45.88, 41.54, 67.35, 2.22, 80.45, 70.82, 31.37, 10.39, 13.69, 16.88, 1.39, 10.55]
    }

    for action in actions_list:
        for i in range(12):
            xhand_exam._hand_command.finger_command[i].position = actions_list[action][i] * math.pi / 180
        xhand_exam.exam_send_command()
        time.sleep(1)

    # Close device
    xhand_exam.exam_close_device()