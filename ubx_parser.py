"""UBX RXM-RAWX parser

This module scans a UBX binary stream, validates checksums, extracts
UBX-RXM-RAWX messages (class=0x02, id=0x15) and decodes their payload
into a Python dict. The RAWX decode uses the common u-blox layout
(best-effort matching ZED-F9P RAWX): header + repeated 32-byte measurement
blocks. If your device differs, adjust offsets/structs in `parse_rawx`.

Usage:
	from ubx_parser import parse_ubx_file
	for msg in parse_ubx_file('path/to/file.ubx'):
		if msg['class']==0x02 and msg['id']==0x15:
			rawx = parse_rawx(msg['payload'])
			print(rawx)

"""
import struct
from typing import Dict, List, Iterator
import numpy as np
from scipy.io import savemat

SYNC1 = 0xB5
SYNC2 = 0x62
RAWX_CLASS = 0x02
RAWX_ID = 0x15


def ubx_checksum(data: bytes) -> bytes:
	"""Compute UBX 8-bit Fletcher checksum (CK_A, CK_B) over data.
	`data` should be bytes starting at message class (i.e. class..payload)."""
	ck_a = 0
	ck_b = 0
	for b in data:
		ck_a = (ck_a + b) & 0xFF
		ck_b = (ck_b + ck_a) & 0xFF
	return bytes((ck_a, ck_b))


def iter_ubx_messages(buf: bytes) -> Iterator[Dict]:
	"""Yield parsed UBX messages found in buffer as dicts:
	{ 'class': int, 'id': int, 'payload': bytes, 'start': int, 'length': int }
	The function is robust to noise and will scan for sync chars.
	"""
	i = 0
	n = len(buf)
	while i + 8 <= n:
		# look for sync
		if buf[i] != SYNC1 or buf[i+1] != SYNC2:
			i += 1
			continue
		if i + 6 > n:
			break
		msg_class = buf[i+2]
		msg_id = buf[i+3]
		msg_len = struct.unpack_from('<H', buf, i+4)[0]
		end = i + 6 + msg_len + 2  # header(6) + payload + ck(2)
		if end > n:
			break
		payload = buf[i+6:i+6+msg_len]
		ck = buf[i+6+msg_len:i+6+msg_len+2]
		calc = ubx_checksum(buf[i+2:i+6+msg_len])
		valid = calc == ck
		yield {
			'start': i,
			'class': msg_class,
			'id': msg_id,
			'length': msg_len,
			'payload': payload,
			'ck': ck,
			'ck_calc': calc,
			'valid': valid,
		}
		i = end


def parse_rawx(payload: bytes) -> Dict:
	"""Decode UBX-RXM-RAWX payload into a dict.

	This implements a commonly used RAWX layout:
	header: rcvTow (double), week (uint16), leapS (int8), numMeas (uint8),
			recStat (uint8), reserved (uint8)
	each measurement block (32 bytes):
		prMes (double), cpMes (double), doMes (float),
		gnssId (uint8), svId (uint8), sigId (uint8), freqId (int8),
		locktime (uint16), cno (uint8), prStdev (uint8), cpStdev (uint8),
		doStdev (uint8), flags (uint16)

	If your device's layout differs, change unpack formats accordingly.
	"""
	out = {}
	off = 0
	plen = len(payload)
	if plen < 8:
		raise ValueError('RAWX payload too short')
	# header
	rcv_tow = struct.unpack_from('<d', payload, off)[0]
	off += 8
	if off + 6 <= plen:
		week = struct.unpack_from('<H', payload, off)[0]
		off += 2
		leap_s = struct.unpack_from('<b', payload, off)[0]
		off += 1
		num_meas = struct.unpack_from('<B', payload, off)[0]
		off += 1
		rec_stat = struct.unpack_from('<B', payload, off)[0]
		off += 1
		version = struct.unpack_from('<B', payload, off)[0]
		off += 1
		reserved = struct.unpack_from('<B', payload, off)[0]
		off += 2
	else:
		# be permissive if header shorter
		week = None
		leap_s = None
		num_meas = 0
		rec_stat = None
		version = None
		reserved = None

	out['rcv_tow'] = rcv_tow
	out['week'] = week
	out['leap_s'] = leap_s
	out['num_meas_reported'] = num_meas
	out['rec_stat'] = rec_stat
	out['version'] = version
	out['reserved'] = reserved
	meas_list: List[Dict] = []

	# per-measurement struct (32 bytes) -- best-effort mapping
	meas_fmt = '<ddfBBBBHBbbbbB'
	meas_size = struct.calcsize(meas_fmt)
	for m in range(num_meas):
		if off + meas_size > plen:
			break
		(pr_mes, cp_mes, do_mes,
		 gnss_id, sv_id, sig_id, freq_id,
		 locktime,
		 cno, pr_stdev, cp_stdev, do_stdev,
		 trkStatus, rez) = struct.unpack_from(meas_fmt, payload, off)
		off += meas_size
		meas_list.append({
			'pr_mes': pr_mes,
			'cp_mes': cp_mes,
			'do_mes': do_mes,
			'gnss_id': gnss_id,
			'sv_id': sv_id,
			'sig_id': sig_id,
			'freq_id': freq_id,
			'locktime': locktime,
			'cno': cno,
			'pr_stdev': pr_stdev,
			'cp_stdev': cp_stdev,
			'do_stdev': do_stdev,
			'trkStatus': trkStatus,
		})

	out['measurements'] = meas_list
	out['raw_trailing'] = payload[off:]
	return out


def parse_ubx_file(path: str) -> List[Dict]:
	"""Read a file and return a list of UBX message dicts found."""
	with open(path, 'rb') as f:
		buf = f.read()
	return list(iter_ubx_messages(buf))


if __name__ == '__main__':
	import sys
	try:
		import tkinter as tk
		from tkinter import filedialog
		root = tk.Tk()
		root.withdraw()
		path = filedialog.askopenfilename(
			title='Select UBX file',
			filetypes=[('UBX files', '*.ubx'), ('All files', '*.*')]
		)
		if not path:
			print('No file selected.')
			sys.exit(1)
	except Exception:
		# fallback to CLI if tkinter not available
		if len(sys.argv) < 2:
			print('Usage: python ubx_parser.py path/to/file.ubx')
			sys.exit(1)
		path = sys.argv[1]

	msgs = parse_ubx_file(path)
	# collect RAWX messages
	rawx_list = []
	for m in msgs:
		if m['class'] == RAWX_CLASS and m['id'] == RAWX_ID and m['valid']:
			rawx = parse_rawx(m['payload'])
			# preserve some UBX metadata
			rawx['ubx_start'] = m['start']
			rawx['ubx_length'] = m['length']
			rawx_list.append(rawx)

	# Build 40 x N matrices for observation data (rows=channels=40, cols=messages)
	# First pass: collect all unique (gnss_id, sv_id) pairs across all messages
	sat_set = set()
	for rawx in rawx_list:
		meas = rawx.get('measurements', [])
		for m in meas:
			gnss_id_val = int(m.get('gnss_id', 0))
			sv_id_val = int(m.get('sv_id', 0))
			sat_set.add((gnss_id_val, sv_id_val))
	
	# Create sorted list of unique satellites and mapping
	sat_list = sorted(list(sat_set))
	sat_to_row = {sat: idx for idx, sat in enumerate(sat_list)}
	n_channels = len(sat_list)  # dynamically set number of channels
	M = len(rawx_list)

	pr_mes = np.zeros((n_channels, M), dtype=np.float64)
	cp_mes = np.zeros((n_channels, M), dtype=np.float64)
	do_mes = np.zeros((n_channels, M), dtype=np.float64)

	gnss_id = np.zeros((n_channels, M), dtype=np.int32)
	sv_id = np.zeros((n_channels, M), dtype=np.int32)
	sig_id = np.zeros((n_channels, M), dtype=np.int32)
	freq_id = np.zeros((n_channels, M), dtype=np.int32)
	locktime = np.zeros((n_channels, M), dtype=np.int32)

	cno = np.zeros((n_channels, M), dtype=np.float64)
	pr_stdev = np.zeros((n_channels, M), dtype=np.float64)
	cp_stdev = np.zeros((n_channels, M), dtype=np.float64)
	do_stdev = np.zeros((n_channels, M), dtype=np.float64)

	trkStatus = np.zeros((n_channels, M), dtype=np.int32)

	# delta arrays (initial value 0)
	pr_mes_delta = np.zeros((n_channels, M), dtype=np.float64)
	cp_mes_delta = np.zeros((n_channels, M), dtype=np.float64)  # wrapped by 2*pi
	do_mes_delta = np.zeros((n_channels, M), dtype=np.float64)
	cno_delta = np.zeros((n_channels, M), dtype=np.float64)

	# per-message header vectors
	rcv_tow = np.zeros((M,), dtype=np.float64)
	week = np.zeros((M,), dtype=np.int32)
	leap_s = np.zeros((M,), dtype=np.int32)
	rec_stat = np.zeros((M,), dtype=np.int32)
	ubx_start = np.zeros((M,), dtype=np.int32)
	ubx_length = np.zeros((M,), dtype=np.int32)

	for j, rawx in enumerate(rawx_list):
		rcv_tow[j] = float(rawx.get('rcv_tow', 0.0))
		week[j] = 0 if rawx.get('week') is None else int(rawx.get('week'))
		leap_s[j] = 0 if rawx.get('leap_s') is None else int(rawx.get('leap_s'))
		rec_stat[j] = 0 if rawx.get('rec_stat') is None else int(rawx.get('rec_stat'))
		ubx_start[j] = int(rawx.get('ubx_start', 0))
		ubx_length[j] = int(rawx.get('ubx_length', 0))

		meas = rawx.get('measurements', [])
		for m in meas:
			gnss_id_val = int(m.get('gnss_id', 0))
			sv_id_val = int(m.get('sv_id', 0))
			sat_key = (gnss_id_val, sv_id_val)
			
			# Get row index for this satellite
			if sat_key in sat_to_row:
				i = sat_to_row[sat_key]
				pr_mes[i, j] = float(m.get('pr_mes', 0.0))
				cp_mes[i, j] = float(m.get('cp_mes', 0.0))
				do_mes[i, j] = float(m.get('do_mes', 0.0))

				gnss_id[i, j] = gnss_id_val
				sv_id[i, j] = sv_id_val
				sig_id[i, j] = int(m.get('sig_id', 0))
				freq_id[i, j] = int(m.get('freq_id', 0))
				locktime[i, j] = int(m.get('locktime', 0))

				cno[i, j] = float(m.get('cno', 0))
				pr_stdev[i, j] = float(m.get('pr_stdev', 0))
				cp_stdev[i, j] = float(m.get('cp_stdev', 0))
				do_stdev[i, j] = float(m.get('do_stdev', 0))

				trkStatus[i, j] = int(m.get('trkStatus', 0))

	# compute deltas (first column remains 0)
	for j in range(1, M):
		# vectorized across channels for efficiency
		pr_mes_delta[:, j] = pr_mes[:, j] - pr_mes[:, j-1]
		do_mes_delta[:, j] = do_mes[:, j] - do_mes[:, j-1]
		cno_delta[:, j] = cno[:, j] - cno[:, j-1]
		cp_mes_delta[:, j] = cp_mes[:, j] - cp_mes[:, j-1]

	out_path = path + '.mat'
	savemat(out_path, {
		'pr_mes': pr_mes,
		'cp_mes': cp_mes,
		'do_mes': do_mes,
		'gnss_id': gnss_id,
		'sv_id': sv_id,
		'sig_id': sig_id,
		'freq_id': freq_id,
		'locktime': locktime,
		'cno': cno,
		'pr_stdev': pr_stdev,
		'cp_stdev': cp_stdev,
		'do_stdev': do_stdev,
		'trkStatus': trkStatus,
		'rcv_tow': rcv_tow,
		'week': week,
		'leap_s': leap_s,
		'rec_stat': rec_stat,
		'ubx_start': ubx_start,
		'ubx_length': ubx_length,
		# deltas
		'pr_mes_delta': pr_mes_delta,
		'cp_mes_delta': cp_mes_delta,
		'do_mes_delta': do_mes_delta,
		'cno_delta': cno_delta,
	})
	print(f'Exported {M} RAWX messages to {out_path}')

