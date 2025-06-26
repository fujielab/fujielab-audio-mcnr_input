"""
OutputCaptureWinを使って5秒間録音し、test_output.wavに保存するプログラム
"""
import time
import numpy as np
import soundfile as sf
from fujielab.audio.mcnr_input._backend.output_capture_win import OutputCaptureWin


def main():
    """メイン関数：5秒間録音してtest_output.wavに保存"""
    print("=== Windows音声出力キャプチャテスト ===")
    
    # OutputCaptureWinのインスタンスを作成
    capture = OutputCaptureWin(sample_rate=16000, channels=2, debug=True)
    
    try:
        # 音声キャプチャを開始
        print("音声キャプチャを開始しています...")
        if not capture.start_audio_capture():
            print("エラー: 音声キャプチャの開始に失敗しました")
            return
        
        print("録音を開始しました。5秒間録音します...")
        
        # 録音データを格納するリスト
        recorded_data = []
        
        # 開始時刻を記録
        start_time = time.time()
        recording_duration = 10.0  # 10秒間録音

        # 10秒間データを収集
        while time.time() - start_time < recording_duration:
            try:
                # 音声データを読み取り
                audio_data = capture.read_audio_capture()
                recorded_data.append(audio_data.data)
                
                # 進捗表示
                elapsed = time.time() - start_time
                print(f"\r録音中... {elapsed:.1f}秒 / {recording_duration}秒", end="", flush=True)
                
            except RuntimeError as e:
                print(f"\nエラー: {e}")
                break
            except KeyboardInterrupt:
                print("\n録音が中断されました")
                break
        
        print(f"\n録音完了！{time.time() - start_time:.1f}秒間録音しました")
        
        # データが収集されている場合、WAVファイルに保存
        if recorded_data:
            try:
                # 全てのデータを結合
                all_samples = np.vstack(recorded_data)
                
                # test_output.wavに保存
                output_filename = "test_output.wav"
                sf.write(output_filename, all_samples, capture.sample_rate)
                
                # 実際の録音時間を計算
                actual_duration = len(all_samples) / capture.sample_rate
                
                print(f"音声データを {output_filename} に保存しました")
                print(f"ファイル情報:")
                print(f"  - サンプリングレート: {capture.sample_rate} Hz")
                print(f"  - チャンネル数: {capture.channels}")
                print(f"  - 録音時間: {actual_duration:.2f} 秒")
                print(f"  - データサイズ: {all_samples.shape}")
                
            except Exception as e:
                print(f"ファイル保存エラー: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("録音データがありません")
    
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 必ずキャプチャを停止
        print("音声キャプチャを停止しています...")
        capture.stop_audio_capture()
        print("完了")


if __name__ == "__main__":
    main()
