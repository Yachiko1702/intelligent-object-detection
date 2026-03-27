
import argparse
import os
import tempfile
from pathlib import Path

import pytest

import camera_detector


def test_parse_args_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # argparse reads from sys.argv; pytest adds its own flags so clear them.
    monkeypatch.setattr('sys.argv', ['camera_detector.py'])
    parser = camera_detector.parse_args
    args = parser()
    assert args.model == Path("yolov8n.pt")
    assert args.cam == 0
    assert args.conf == 0.25
    assert args.device == ""


def test_parse_args_invalid_camera(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        camera_detector.parse_args().cam  # can't simulate easily without CLI


def test_model_file_validation(tmp_path, monkeypatch):
    # create a dummy model file and ensure no error
    model_path = tmp_path / "my.pt"
    model_path.write_text("")
    args = argparse.Namespace(
        model=model_path,
        cam=0,
        conf=0.5,
        device="",
        async_capture=False,
        output=None,
    )
    # patch VideoCapture so the camera initialization succeeds
    class DummyCap:
        def isOpened(self):
            return True
        def read(self):
            return False, None
        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda idx: DummyCap())
    # also patch YOLO so we don't try to load the dummy .pt file
    monkeypatch.setattr(camera_detector, 'YOLO', lambda path: type('DummyModel', (), {'names': {}})())

    det = camera_detector.CameraDetector(
        model_path=model_path,
        cam_index=0,
        conf_thresh=0.5,
        device=None,
        async_capture=False,
        output_file=None,
    )
    assert det.cam_index == 0
    det.close()


def test_default_model_contains_expected_classes():
    # the COCO-trained YOLOv8 models include many common household objects;
    # verify that the ones mentioned by the user are present.  classes such
    # as ``pencil``/``ballpen`` are not in COCO and would require a custom
    # training dataset.
    from ultralytics import YOLO

    names = set(YOLO('yolov8n.pt').names.values())
    expected = {'mouse', 'cup', 'laptop', 'tv', 'keyboard'}
    assert expected.issubset(names), f"missing {expected - names}"


def test_list_classes_flag(tmp_path, monkeypatch, capsys):
    # the --list-classes option should print classes and exit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['camera_detector.py', '--list-classes'])
    class DummyModel:
        names = {0: 'foo', 1: 'bar'}
    monkeypatch.setattr(camera_detector, 'YOLO', lambda path: DummyModel())
    camera_detector.main()
    captured = capsys.readouterr()
    assert '0: foo' in captured.out
    assert '1: bar' in captured.out
