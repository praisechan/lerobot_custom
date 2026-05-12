import threading

import torch

from greenctx_extension import load_greenctx_extension
from pi0_infer import (
    Pi0Inference,
    decoder_model,
    encoder_model,
    transformer_decoder,
    transformer_encoder,
    vision_encoder,
)


def greenctx_encoder_model(weights, buffers, num_views):
    encoder_seq_len = buffers["encoder_x"].shape[0]
    torch.cuda.nvtx.range_push("GreenCtx Vision Encoder")
    vision_encoder(weights, buffers, num_views)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("GreenCtx Transformer Encoder")
    transformer_encoder(weights, buffers, encoder_seq_len)
    torch.cuda.nvtx.range_pop()


def greenctx_decoder_model(weights, buffers, encoder_seq_len):
    torch.cuda.nvtx.range_push("GreenCtx Transformer Decoder")
    transformer_decoder(weights, buffers, encoder_seq_len)
    torch.cuda.nvtx.range_pop()


class Pi0GreenContextInference(Pi0Inference):
    """Pi0Inference variant that replays encoder/decoder graphs on GreenContext streams."""

    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        encoder_sms,
        decoder_sms,
        device=None,
        verbose=True,
    ):
        if device is None:
            device = torch.cuda.current_device()
        self.device = int(device)
        torch.cuda.set_device(self.device)
        torch.cuda.init()

        super().__init__(checkpoint, num_views=num_views, chunk_size=chunk_size)
        torch.cuda.synchronize()

        ext = load_greenctx_extension()
        self.greenctx = ext.GreenContextPair(
            self.device, int(encoder_sms), int(decoder_sms), bool(verbose)
        )
        self.green_encoder_stream, self.green_decoder_stream = self._make_external_streams()
        self.full_sm_encoder_stream = torch.cuda.Stream()
        self.full_sm_decoder_stream = torch.cuda.Stream()
        self.record_greenctx_graphs()

    def record_greenctx_graphs(self):
        for _ in range(3):
            self.greenctx.set_encoder_current()
            with torch.cuda.stream(self.green_encoder_stream):
                encoder_model(self.weights, self.buffers, self.num_views)
            self.greenctx.synchronize_encoder()

            self.greenctx.set_decoder_current()
            with torch.cuda.stream(self.green_decoder_stream):
                decoder_model(self.weights, self.buffers, self.encoder_seq_len)
            self.greenctx.synchronize_decoder()

        self.greenctx.set_encoder_current()
        self.green_encoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.green_encoder_stream):
            self.green_encoder_graph.capture_begin()
            encoder_model(self.weights, self.buffers, self.num_views)
            self.green_encoder_graph.capture_end()
        self.greenctx.synchronize_encoder()

        self.greenctx.set_decoder_current()
        self.green_decoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.green_decoder_stream):
            self.green_decoder_graph.capture_begin()
            decoder_model(self.weights, self.buffers, self.encoder_seq_len)
            self.green_decoder_graph.capture_end()
        self.greenctx.synchronize_decoder()
        self.greenctx.restore_primary_current()

    def _make_external_streams(self):
        self.greenctx.set_encoder_current()
        encoder_stream = torch.cuda.ExternalStream(
            self.greenctx.encoder_stream_ptr(), device=self.device
        )
        self.greenctx.set_decoder_current()
        decoder_stream = torch.cuda.ExternalStream(
            self.greenctx.decoder_stream_ptr(), device=self.device
        )
        self.greenctx.restore_primary_current()
        return encoder_stream, decoder_stream

    def greenctx_info(self):
        return dict(self.greenctx.info())

    def prepare_inputs(
        self,
        observation_images_normalized,
        observation_state_normalized,
        diffusion_noise,
    ):
        self.greenctx.restore_primary_current()
        self.buffers["observation_images_normalized"].copy_(observation_images_normalized)
        self.buffers["observation_state_normalized"].copy_(observation_state_normalized)
        self.buffers["diffusion_noise"].copy_(diffusion_noise)

    def forward_concurrent_cached_streams(
        self,
        observation_images_normalized,
        observation_state_normalized,
        diffusion_noise,
    ):
        self.prepare_inputs(
            observation_images_normalized,
            observation_state_normalized,
            diffusion_noise,
        )
        start_event = torch.cuda.Event()
        start_event.record()
        with torch.cuda.stream(self.full_sm_encoder_stream):
            self.full_sm_encoder_stream.wait_event(start_event)
            self.encoder_graph.replay()
        with torch.cuda.stream(self.full_sm_decoder_stream):
            self.full_sm_decoder_stream.wait_event(start_event)
            self.decoder_graph.replay()
        return self.buffers["diffusion_noise"]

    def replay_encoder_greenctx(self, synchronize=True, profile_markers=False):
        if profile_markers:
            torch.cuda.nvtx.range_push("GreenCtx Encoder Only")
        try:
            self.greenctx.set_encoder_current()
            with torch.cuda.stream(self.green_encoder_stream):
                self.green_encoder_graph.replay()
            if synchronize:
                self.greenctx.synchronize_encoder()
        finally:
            self.greenctx.restore_primary_current()
            if profile_markers:
                torch.cuda.nvtx.range_pop()

    def replay_decoder_greenctx(self, synchronize=True, profile_markers=False):
        if profile_markers:
            torch.cuda.nvtx.range_push("GreenCtx Decoder Only")
        try:
            self.greenctx.set_decoder_current()
            with torch.cuda.stream(self.green_decoder_stream):
                self.green_decoder_graph.replay()
            if synchronize:
                self.greenctx.synchronize_decoder()
        finally:
            self.greenctx.restore_primary_current()
            if profile_markers:
                torch.cuda.nvtx.range_pop()

    def run_encoder_direct_greenctx(self, synchronize=True, profile_markers=False):
        if profile_markers:
            torch.cuda.nvtx.range_push("GreenCtx Direct Encoder")
        try:
            self.greenctx.set_encoder_current()
            with torch.cuda.stream(self.green_encoder_stream):
                greenctx_encoder_model(self.weights, self.buffers, self.num_views)
            if synchronize:
                self.greenctx.synchronize_encoder()
        finally:
            self.greenctx.restore_primary_current()
            if profile_markers:
                torch.cuda.nvtx.range_pop()

    def run_decoder_direct_greenctx(self, synchronize=True, profile_markers=False):
        if profile_markers:
            torch.cuda.nvtx.range_push("GreenCtx Direct Decoder")
        try:
            self.greenctx.set_decoder_current()
            with torch.cuda.stream(self.green_decoder_stream):
                greenctx_decoder_model(self.weights, self.buffers, self.encoder_seq_len)
            if synchronize:
                self.greenctx.synchronize_decoder()
        finally:
            self.greenctx.restore_primary_current()
            if profile_markers:
                torch.cuda.nvtx.range_pop()

    def replay_concurrent_greenctx(self, profile_markers=False):
        errors = []
        barrier = threading.Barrier(3)

        def run_encoder():
            pushed = False
            try:
                self.greenctx.set_encoder_current()
                barrier.wait(timeout=30.0)
                if profile_markers:
                    torch.cuda.nvtx.range_push("GreenCtx Concurrent Encoder")
                    pushed = True
                with torch.cuda.stream(self.green_encoder_stream):
                    self.green_encoder_graph.replay()
                self.greenctx.synchronize_encoder()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()
            finally:
                self.greenctx.restore_primary_current()
                if pushed:
                    torch.cuda.nvtx.range_pop()

        def run_decoder():
            pushed = False
            try:
                self.greenctx.set_decoder_current()
                barrier.wait(timeout=30.0)
                if profile_markers:
                    torch.cuda.nvtx.range_push("GreenCtx Concurrent Decoder")
                    pushed = True
                with torch.cuda.stream(self.green_decoder_stream):
                    self.green_decoder_graph.replay()
                self.greenctx.synchronize_decoder()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()
            finally:
                self.greenctx.restore_primary_current()
                if pushed:
                    torch.cuda.nvtx.range_pop()

        if profile_markers:
            torch.cuda.nvtx.range_push("GreenCtx Concurrent Total")
        try:
            encoder_thread = threading.Thread(target=run_encoder, name="greenctx-encoder")
            decoder_thread = threading.Thread(target=run_decoder, name="greenctx-decoder")
            encoder_thread.start()
            decoder_thread.start()
            try:
                barrier.wait(timeout=30.0)
            finally:
                encoder_thread.join()
                decoder_thread.join()
            if errors:
                raise errors[0]
        finally:
            self.greenctx.restore_primary_current()
            if profile_markers:
                torch.cuda.nvtx.range_pop()

    def run_concurrent_direct_greenctx(self, profile_markers=False):
        errors = []
        barrier = threading.Barrier(3)

        def run_encoder():
            pushed = False
            try:
                self.greenctx.set_encoder_current()
                barrier.wait(timeout=30.0)
                if profile_markers:
                    torch.cuda.nvtx.range_push("GreenCtx Direct Concurrent Encoder")
                    pushed = True
                with torch.cuda.stream(self.green_encoder_stream):
                    greenctx_encoder_model(self.weights, self.buffers, self.num_views)
                self.greenctx.synchronize_encoder()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()
            finally:
                self.greenctx.restore_primary_current()
                if pushed:
                    torch.cuda.nvtx.range_pop()

        def run_decoder():
            pushed = False
            try:
                self.greenctx.set_decoder_current()
                barrier.wait(timeout=30.0)
                if profile_markers:
                    torch.cuda.nvtx.range_push("GreenCtx Direct Concurrent Decoder")
                    pushed = True
                with torch.cuda.stream(self.green_decoder_stream):
                    greenctx_decoder_model(self.weights, self.buffers, self.encoder_seq_len)
                self.greenctx.synchronize_decoder()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()
            finally:
                self.greenctx.restore_primary_current()
                if pushed:
                    torch.cuda.nvtx.range_pop()

        if profile_markers:
            torch.cuda.nvtx.range_push("GreenCtx Direct Concurrent Total")
        try:
            encoder_thread = threading.Thread(target=run_encoder, name="greenctx-direct-encoder")
            decoder_thread = threading.Thread(target=run_decoder, name="greenctx-direct-decoder")
            encoder_thread.start()
            decoder_thread.start()
            try:
                barrier.wait(timeout=30.0)
            finally:
                encoder_thread.join()
                decoder_thread.join()
            if errors:
                raise errors[0]
        finally:
            self.greenctx.restore_primary_current()
            if profile_markers:
                torch.cuda.nvtx.range_pop()

    def synchronize_greenctx(self):
        self.greenctx.synchronize_both()
