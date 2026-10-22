import base64
import json
from pathlib import Path
import socket
import struct
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocol as p


class ProtocolTests(unittest.TestCase):
    def test_fragmented_framing_and_multiple_packets(self):
        left, right = socket.socketpair()
        try:
            left.settimeout(.2)
            right.settimeout(.2)
            raw=json.dumps({'version':1,'type':'ping'}).encode()
            left.sendall(struct.pack('!I',len(raw))[:2])
            left.sendall(struct.pack('!I',len(raw))[2:]+raw)
            p.send_packet(left, {'version':1,'type':'hello'})
            self.assertEqual(p.recv_packet(right)['type'],'ping')
            self.assertEqual(p.recv_packet(right)['type'],'hello')
        finally:
            left.close(); right.close()

    def test_rejects_oversize_nonfinite_duplicate_and_wrong_version(self):
        for data in (b'{"version":1,"x":NaN}',b'{"version":1,"version":1}', b'{"version":true}', b'[]'):
            left,right=socket.socketpair()
            try:
                left.sendall(struct.pack('!I',len(data))+data)
                with self.assertRaises(p.ProtocolError): p.recv_packet(right)
            finally:
                left.close();right.close()
        left,right=socket.socketpair()
        try:
            left.sendall(struct.pack('!I',p.MAX_PACKET_BYTES+1))
            with self.assertRaises(p.ProtocolError):p.recv_packet(right)
        finally:
            left.close();right.close()

    def test_valid_response_must_belong_to_same_observation_and_model(self):
        request={'session_id':uuid.uuid4().hex,'request_id':4,'capture_seq':3,'capture_monotonic':200.5}
        response={'version':1,'type':'actions',**request,'fps':15,'model_id':'a','actions':[[0.0]*8]*50}
        p.validate_actions_response(response,request,'a')
        for field,value in [('session_id',uuid.uuid4().hex),('capture_seq',2),('request_id',3),('capture_monotonic',200.0),('model_id','b')]:
            with self.subTest(field=field), self.assertRaises(p.ProtocolError):
                p.validate_actions_response({**response,field:value},request,'a')
        with self.assertRaises(p.ProtocolError):
            p.validate_actions_response({**response,'actions':[[float('nan')]*8]},request,'a')

    def test_rgb_jpeg_roundtrip_preserves_channel_order_and_shape(self):
        try:
            import numpy as np
        except ImportError as e: self.skipTest(str(e))
        image=np.zeros((480,640,3),dtype=np.uint8)
        image[:]=[240,20,5]
        decoded=p.decode_image(p.encode_image(image))
        self.assertEqual(decoded.shape,(480,640,3))
        self.assertGreater(float(decoded[:,:,0].mean()),230)
        self.assertLess(float(decoded[:,:,2].mean()),15)
        with self.assertRaises(p.ProtocolError):p.encode_image(image.astype('float32'))
        with self.assertRaises(p.ProtocolError):p.decode_image({'encoding':'jpeg','width':640,'height':480,'data':'!!!!'})


if __name__=='__main__':unittest.main()
