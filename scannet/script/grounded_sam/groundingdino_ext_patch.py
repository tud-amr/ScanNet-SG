"""
GroundingDINO: if `groundingdino._C` fails to import, upstream leaves `_C` undefined but
still runs the CUDA autograd path when tensors are on GPU (`name '_C' is not defined`).

Call `apply_groundingdino_ms_deform_attn_patch()` after GroundingDINO is on `sys.path` and
before loading weights. This replaces `MultiScaleDeformableAttention.forward` only when
`_C` is missing, using the same PyTorch fallback upstream already ships for CPU.
"""

from __future__ import annotations

import warnings


def apply_groundingdino_ms_deform_attn_patch() -> None:
    import torch

    import groundingdino.models.GroundingDINO.ms_deform_attn as mda

    if not hasattr(mda, "_C"):
        mda._C = None
    if mda._C is not None:
        return

    cls = mda.MultiScaleDeformableAttention

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        if mda._C is not None and torch.cuda.is_available() and value.is_cuda:
            halffloat = False
            if value.dtype == torch.float16:
                halffloat = True
                value = value.float()
                sampling_locations = sampling_locations.float()
                attention_weights = attention_weights.float()

            output = mda.MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

            if halffloat:
                output = output.half()
        else:
            output = mda.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    cls.forward = forward
    warnings.warn(
        "groundingdino._C is not available; using PyTorch multi-scale deformable attention "
        "on GPU (slower). Install/build GroundingDINO with CUDA extensions to speed this up.",
        stacklevel=2,
    )
