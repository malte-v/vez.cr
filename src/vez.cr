require "vulkan"

@[Link("vez")]
lib Vez
  alias FramebufferT = Void
  alias MemoryFlags = Vk::Flags
  alias PipelineT = Void
  alias SwapchainT = Void
  alias VertexInputFormatT = Void

  enum BaseType
    Bool   = 0
    Char   = 1
    Int    = 2
    Uint   = 3
    Uint64 = 4
    Half   = 5
    Float  = 6
    Double = 7
    Struct = 8
  end
  enum PipelineResourceType
    Input                =  0
    Output               =  1
    Sampler              =  2
    CombinedImageSampler =  3
    SampledImage         =  4
    StorageImage         =  5
    UniformTexelBuffer   =  6
    StorageTexelBuffer   =  7
    UniformBuffer        =  8
    StorageBuffer        =  9
    InputAttachment      = 10
    PushConstantBuffer   = 11
  end

  fun allocate_command_buffers = vezAllocateCommandBuffers(device : Vk::Device, allocate_info : CommandBufferAllocateInfo*, command_buffers : Vk::CommandBuffer*) : Vk::Result
  fun begin_command_buffer = vezBeginCommandBuffer(command_buffer : Vk::CommandBuffer, flags : Vk::CommandBufferUsageFlags) : Vk::Result
  fun buffer_sub_data = vezBufferSubData(device : Vk::Device, buffer : Vk::Buffer, offset : Vk::DeviceSize, size : Vk::DeviceSize, data : Void*) : Vk::Result
  fun cmd_begin_render_pass = vezCmdBeginRenderPass(begin_info : RenderPassBeginInfo*)
  fun cmd_bind_buffer = vezCmdBindBuffer(buffer : Vk::Buffer, offset : Vk::DeviceSize, range : Vk::DeviceSize, set : LibC::UInt, binding : LibC::UInt, array_element : LibC::UInt)
  fun cmd_bind_buffer_view = vezCmdBindBufferView(buffer_view : Vk::BufferView, set : LibC::UInt, binding : LibC::UInt, array_element : LibC::UInt)
  fun cmd_bind_image_view = vezCmdBindImageView(image_view : Vk::ImageView, sampler : Vk::Sampler, set : LibC::UInt, binding : LibC::UInt, array_element : LibC::UInt)
  fun cmd_bind_index_buffer = vezCmdBindIndexBuffer(buffer : Vk::Buffer, offset : Vk::DeviceSize, index_type : Vk::IndexType)
  fun cmd_bind_pipeline = vezCmdBindPipeline(pipeline : Pipeline)
  fun cmd_bind_sampler = vezCmdBindSampler(sampler : Vk::Sampler, set : LibC::UInt, binding : LibC::UInt, array_element : LibC::UInt)
  fun cmd_bind_vertex_buffers = vezCmdBindVertexBuffers(first_binding : LibC::UInt, binding_count : LibC::UInt, buffers : Vk::Buffer*, offsets : Vk::DeviceSize*)
  fun cmd_blit_image = vezCmdBlitImage(src_image : Vk::Image, dst_image : Vk::Image, region_count : LibC::UInt, regions : ImageBlit*, filter : Vk::Filter)
  fun cmd_clear_attachments = vezCmdClearAttachments(attachment_count : LibC::UInt, attachments : ClearAttachment*, rect_count : LibC::UInt, rects : Vk::ClearRect*)
  fun cmd_clear_color_image = vezCmdClearColorImage(image : Vk::Image, color : Vk::ClearColorValue*, range_count : LibC::UInt, ranges : ImageSubresourceRange*)
  fun cmd_clear_depth_stencil_image = vezCmdClearDepthStencilImage(image : Vk::Image, depth_stencil : Vk::ClearDepthStencilValue*, range_count : LibC::UInt, ranges : ImageSubresourceRange*)
  fun cmd_copy_buffer = vezCmdCopyBuffer(src_buffer : Vk::Buffer, dst_buffer : Vk::Buffer, region_count : LibC::UInt, regions : BufferCopy*)
  fun cmd_copy_buffer_to_image = vezCmdCopyBufferToImage(src_buffer : Vk::Buffer, dst_image : Vk::Image, region_count : LibC::UInt, regions : BufferImageCopy*)
  fun cmd_copy_image = vezCmdCopyImage(src_image : Vk::Image, dst_image : Vk::Image, region_count : LibC::UInt, regions : ImageCopy*)
  fun cmd_copy_image_to_buffer = vezCmdCopyImageToBuffer(src_image : Vk::Image, dst_buffer : Vk::Buffer, region_count : LibC::UInt, regions : BufferImageCopy*)
  fun cmd_dispatch = vezCmdDispatch(group_count_x : LibC::UInt, group_count_y : LibC::UInt, group_count_z : LibC::UInt)
  fun cmd_dispatch_indirect = vezCmdDispatchIndirect(buffer : Vk::Buffer, offset : Vk::DeviceSize)
  fun cmd_draw = vezCmdDraw(vertex_count : LibC::UInt, instance_count : LibC::UInt, first_vertex : LibC::UInt, first_instance : LibC::UInt)
  fun cmd_draw_indexed = vezCmdDrawIndexed(index_count : LibC::UInt, instance_count : LibC::UInt, first_index : LibC::UInt, vertex_offset : LibC::Int, first_instance : LibC::UInt)
  fun cmd_draw_indexed_indirect = vezCmdDrawIndexedIndirect(buffer : Vk::Buffer, offset : Vk::DeviceSize, draw_count : LibC::UInt, stride : LibC::UInt)
  fun cmd_draw_indirect = vezCmdDrawIndirect(buffer : Vk::Buffer, offset : Vk::DeviceSize, draw_count : LibC::UInt, stride : LibC::UInt)
  fun cmd_end_render_pass = vezCmdEndRenderPass
  fun cmd_fill_buffer = vezCmdFillBuffer(dst_buffer : Vk::Buffer, dst_offset : Vk::DeviceSize, size : Vk::DeviceSize, data : LibC::UInt)
  fun cmd_next_subpass = vezCmdNextSubpass
  fun cmd_push_constants = vezCmdPushConstants(offset : LibC::UInt, size : LibC::UInt, values : Void*)
  fun cmd_reset_event = vezCmdResetEvent(event : Vk::Event, stage_mask : Vk::PipelineStageFlags)
  fun cmd_resolve_image = vezCmdResolveImage(src_image : Vk::Image, dst_image : Vk::Image, region_count : LibC::UInt, regions : ImageResolve*)
  fun cmd_set_blend_constants = vezCmdSetBlendConstants(blend_constants : LibC::Float[4])
  fun cmd_set_color_blend_state = vezCmdSetColorBlendState(state_info : ColorBlendState*)
  fun cmd_set_depth_bias = vezCmdSetDepthBias(depth_bias_constant_factor : LibC::Float, depth_bias_clamp : LibC::Float, depth_bias_slope_factor : LibC::Float)
  fun cmd_set_depth_bounds = vezCmdSetDepthBounds(min_depth_bounds : LibC::Float, max_depth_bounds : LibC::Float)
  fun cmd_set_depth_stencil_state = vezCmdSetDepthStencilState(state_info : DepthStencilState*)
  fun cmd_set_event = vezCmdSetEvent(event : Vk::Event, stage_mask : Vk::PipelineStageFlags)
  fun cmd_set_input_assembly_state = vezCmdSetInputAssemblyState(state_info : InputAssemblyState*)
  fun cmd_set_line_width = vezCmdSetLineWidth(line_width : LibC::Float)
  fun cmd_set_multisample_state = vezCmdSetMultisampleState(state_info : MultisampleState*)
  fun cmd_set_rasterization_state = vezCmdSetRasterizationState(state_info : RasterizationState*)
  fun cmd_set_scissor = vezCmdSetScissor(first_scissor : LibC::UInt, scissor_count : LibC::UInt, scissors : Vk::Rect2D*)
  fun cmd_set_stencil_compare_mask = vezCmdSetStencilCompareMask(face_mask : Vk::StencilFaceFlags, compare_mask : LibC::UInt)
  fun cmd_set_stencil_reference = vezCmdSetStencilReference(face_mask : Vk::StencilFaceFlags, reference : LibC::UInt)
  fun cmd_set_stencil_write_mask = vezCmdSetStencilWriteMask(face_mask : Vk::StencilFaceFlags, write_mask : LibC::UInt)
  fun cmd_set_vertex_input_format = vezCmdSetVertexInputFormat(format : VertexInputFormat)
  fun cmd_set_viewport = vezCmdSetViewport(first_viewport : LibC::UInt, viewport_count : LibC::UInt, viewports : Vk::Viewport*)
  fun cmd_set_viewport_state = vezCmdSetViewportState(viewport_count : LibC::UInt)
  fun cmd_update_buffer = vezCmdUpdateBuffer(dst_buffer : Vk::Buffer, dst_offset : Vk::DeviceSize, data_size : Vk::DeviceSize, data : Void*)
  fun create_buffer = vezCreateBuffer(device : Vk::Device, mem_flags : MemoryFlags, create_info : BufferCreateInfo*, buffer : Vk::Buffer*) : Vk::Result
  fun create_buffer_view = vezCreateBufferView(device : Vk::Device, create_info : BufferViewCreateInfo*, view : Vk::BufferView*) : Vk::Result
  fun create_compute_pipeline = vezCreateComputePipeline(device : Vk::Device, create_info : ComputePipelineCreateInfo*, pipeline : Pipeline*) : Vk::Result
  fun create_device = vezCreateDevice(physical_device : Vk::PhysicalDevice, create_info : DeviceCreateInfo*, device : Vk::Device*) : Vk::Result
  fun create_event = vezCreateEvent(device : Vk::Device, event : Vk::Event*) : Vk::Result
  fun create_framebuffer = vezCreateFramebuffer(device : Vk::Device, create_info : FramebufferCreateInfo*, framebuffer : Framebuffer*) : Vk::Result
  fun create_graphics_pipeline = vezCreateGraphicsPipeline(device : Vk::Device, create_info : GraphicsPipelineCreateInfo*, pipeline : Pipeline*) : Vk::Result
  fun create_image = vezCreateImage(device : Vk::Device, mem_flags : MemoryFlags, create_info : ImageCreateInfo*, image : Vk::Image*) : Vk::Result
  fun create_image_view = vezCreateImageView(device : Vk::Device, create_info : ImageViewCreateInfo*, view : Vk::ImageView*) : Vk::Result
  fun create_instance = vezCreateInstance(create_info : InstanceCreateInfo*, instance : Vk::Instance*) : Vk::Result
  fun create_query_pool = vezCreateQueryPool(device : Vk::Device, create_info : QueryPoolCreateInfo*, query_pool : Vk::QueryPool*) : Vk::Result
  fun create_sampler = vezCreateSampler(device : Vk::Device, create_info : SamplerCreateInfo*, sampler : Vk::Sampler*) : Vk::Result
  fun create_shader_module = vezCreateShaderModule(device : Vk::Device, create_info : ShaderModuleCreateInfo*, shader_module : Vk::ShaderModule*) : Vk::Result
  fun create_swapchain = vezCreateSwapchain(device : Vk::Device, create_info : SwapchainCreateInfo*, swapchain : Swapchain*) : Vk::Result
  fun create_vertex_input_format = vezCreateVertexInputFormat(device : Vk::Device, create_info : VertexInputFormatCreateInfo*, format : VertexInputFormat*) : Vk::Result
  fun destroy_buffer = vezDestroyBuffer(device : Vk::Device, buffer : Vk::Buffer)
  fun destroy_buffer_view = vezDestroyBufferView(device : Vk::Device, buffer_view : Vk::BufferView)
  fun destroy_device = vezDestroyDevice(device : Vk::Device)
  fun destroy_event = vezDestroyEvent(device : Vk::Device, event : Vk::Event)
  fun destroy_fence = vezDestroyFence(device : Vk::Device, fence : Vk::Fence)
  fun destroy_framebuffer = vezDestroyFramebuffer(device : Vk::Device, framebuffer : Framebuffer)
  fun destroy_image = vezDestroyImage(device : Vk::Device, image : Vk::Image)
  fun destroy_image_view = vezDestroyImageView(device : Vk::Device, image_view : Vk::ImageView)
  fun destroy_instance = vezDestroyInstance(instance : Vk::Instance)
  fun destroy_pipeline = vezDestroyPipeline(device : Vk::Device, pipeline : Pipeline)
  fun destroy_query_pool = vezDestroyQueryPool(device : Vk::Device, query_pool : Vk::QueryPool)
  fun destroy_sampler = vezDestroySampler(device : Vk::Device, sampler : Vk::Sampler)
  fun destroy_semaphore = vezDestroySemaphore(device : Vk::Device, semaphore : Vk::Semaphore)
  fun destroy_shader_module = vezDestroyShaderModule(device : Vk::Device, shader_module : Vk::ShaderModule)
  fun destroy_swapchain = vezDestroySwapchain(device : Vk::Device, swapchain : Swapchain)
  fun destroy_vertex_input_format = vezDestroyVertexInputFormat(device : Vk::Device, format : VertexInputFormat)
  fun device_wait_idle = vezDeviceWaitIdle(device : Vk::Device) : Vk::Result
  fun end_command_buffer = vezEndCommandBuffer : Vk::Result
  fun enumerate_device_extension_properties = vezEnumerateDeviceExtensionProperties(physical_device : Vk::PhysicalDevice, layer_name : LibC::Char*, property_count : LibC::UInt*, properties : Vk::ExtensionProperties*) : Vk::Result
  fun enumerate_device_layer_properties = vezEnumerateDeviceLayerProperties(physical_device : Vk::PhysicalDevice, property_count : LibC::UInt*, properties : Vk::LayerProperties*) : Vk::Result
  fun enumerate_instance_extension_properties = vezEnumerateInstanceExtensionProperties(layer_name : LibC::Char*, property_count : LibC::UInt*, properties : Vk::ExtensionProperties*) : Vk::Result
  fun enumerate_instance_layer_properties = vezEnumerateInstanceLayerProperties(property_count : LibC::UInt*, properties : Vk::LayerProperties*) : Vk::Result
  fun enumerate_physical_devices = vezEnumeratePhysicalDevices(instance : Vk::Instance, physical_device_count : LibC::UInt*, physical_devices : Vk::PhysicalDevice*) : Vk::Result
  fun enumerate_pipeline_resources = vezEnumeratePipelineResources(pipeline : Pipeline, resource_count : LibC::UInt*, resources : PipelineResource*) : Vk::Result
  fun flush_mapped_buffer_ranges = vezFlushMappedBufferRanges(device : Vk::Device, buffer_range_count : LibC::UInt, buffer_ranges : MappedBufferRange*) : Vk::Result
  fun free_command_buffers = vezFreeCommandBuffers(device : Vk::Device, command_buffer_count : LibC::UInt, command_buffers : Vk::CommandBuffer*)
  fun get_device_compute_queue = vezGetDeviceComputeQueue(device : Vk::Device, queue_index : LibC::UInt, queue : Vk::Queue*)
  fun get_device_graphics_queue = vezGetDeviceGraphicsQueue(device : Vk::Device, queue_index : LibC::UInt, queue : Vk::Queue*)
  fun get_device_queue = vezGetDeviceQueue(device : Vk::Device, queue_family_index : LibC::UInt, queue_index : LibC::UInt, queue : Vk::Queue*)
  fun get_device_transfer_queue = vezGetDeviceTransferQueue(device : Vk::Device, queue_index : LibC::UInt, queue : Vk::Queue*)
  fun get_event_status = vezGetEventStatus(device : Vk::Device, event : Vk::Event) : Vk::Result
  fun get_fence_status = vezGetFenceStatus(device : Vk::Device, fence : Vk::Fence) : Vk::Result
  fun get_physical_device_features = vezGetPhysicalDeviceFeatures(physical_device : Vk::PhysicalDevice, features : Vk::PhysicalDeviceFeatures*)
  fun get_physical_device_format_properties = vezGetPhysicalDeviceFormatProperties(physical_device : Vk::PhysicalDevice, format : Vk::Format, format_properties : Vk::FormatProperties*)
  fun get_physical_device_image_format_properties = vezGetPhysicalDeviceImageFormatProperties(physical_device : Vk::PhysicalDevice, format : Vk::Format, type : Vk::ImageType, tiling : Vk::ImageTiling, usage : Vk::ImageUsageFlags, flags : Vk::ImageCreateFlags, image_format_properties : Vk::ImageFormatProperties*) : Vk::Result
  fun get_physical_device_present_support = vezGetPhysicalDevicePresentSupport(physical_device : Vk::PhysicalDevice, queue_family_index : LibC::UInt, surface : Vk::SurfaceKHR, supported : Vk::Bool32*) : Vk::Result
  fun get_physical_device_properties = vezGetPhysicalDeviceProperties(physical_device : Vk::PhysicalDevice, properties : Vk::PhysicalDeviceProperties*)
  fun get_physical_device_queue_family_properties = vezGetPhysicalDeviceQueueFamilyProperties(physical_device : Vk::PhysicalDevice, queue_family_property_count : LibC::UInt*, queue_family_properties : Vk::QueueFamilyProperties*)
  fun get_physical_device_surface_formats = vezGetPhysicalDeviceSurfaceFormats(physical_device : Vk::PhysicalDevice, surface : Vk::SurfaceKHR, surface_format_count : LibC::UInt*, surface_formats : Vk::SurfaceFormatKHR*) : Vk::Result
  fun get_pipeline_resource = vezGetPipelineResource(pipeline : Pipeline, name : LibC::Char*, resource : PipelineResource*) : Vk::Result
  fun get_query_pool_results = vezGetQueryPoolResults(device : Vk::Device, query_pool : Vk::QueryPool, first_query : LibC::UInt, query_count : LibC::UInt, data_size : LibC::SizeT, data : Void*, stride : Vk::DeviceSize, flags : Vk::QueryResultFlags) : Vk::Result
  fun get_shader_module_binary = vezGetShaderModuleBinary(shader_module : Vk::ShaderModule, length : LibC::UInt*, binary : LibC::UInt*) : Vk::Result
  fun get_shader_module_info_log = vezGetShaderModuleInfoLog(shader_module : Vk::ShaderModule, length : LibC::UInt*, info_log : LibC::Char*)
  fun get_swapchain_surface_format = vezGetSwapchainSurfaceFormat(swapchain : Swapchain, format : Vk::SurfaceFormatKHR*)
  fun image_sub_data = vezImageSubData(device : Vk::Device, image : Vk::Image, sub_data_info : ImageSubDataInfo*, data : Void*) : Vk::Result
  fun invalidate_mapped_buffer_ranges = vezInvalidateMappedBufferRanges(device : Vk::Device, buffer_range_count : LibC::UInt, buffer_ranges : MappedBufferRange*) : Vk::Result
  fun map_buffer = vezMapBuffer(device : Vk::Device, buffer : Vk::Buffer, offset : Vk::DeviceSize, size : Vk::DeviceSize, data : Void**) : Vk::Result
  fun queue_present = vezQueuePresent(queue : Vk::Queue, present_info : PresentInfo*) : Vk::Result
  fun queue_submit = vezQueueSubmit(queue : Vk::Queue, submit_count : LibC::UInt, submits : SubmitInfo*, fence : Vk::Fence*) : Vk::Result
  fun queue_wait_idle = vezQueueWaitIdle(queue : Vk::Queue) : Vk::Result
  fun reset_command_buffer = vezResetCommandBuffer(command_buffer : Vk::CommandBuffer) : Vk::Result
  fun reset_event = vezResetEvent(device : Vk::Device, event : Vk::Event) : Vk::Result
  fun set_event = vezSetEvent(device : Vk::Device, event : Vk::Event) : Vk::Result
  fun swapchain_set_v_sync = vezSwapchainSetVSync(swapchain : Swapchain, enabled : Vk::Bool32) : Vk::Result
  fun unmap_buffer = vezUnmapBuffer(device : Vk::Device, buffer : Vk::Buffer)
  fun wait_for_fences = vezWaitForFences(device : Vk::Device, fence_count : LibC::UInt, fences : Vk::Fence*, wait_all : Vk::Bool32, timeout : LibC::ULong) : Vk::Result

  struct ApplicationInfo
    next : Void*
    application_name : LibC::Char*
    application_version : LibC::UInt
    engine_name : LibC::Char*
    engine_version : LibC::UInt
  end

  struct AttachmentInfo
    load_op : Vk::AttachmentLoadOp
    store_op : Vk::AttachmentStoreOp
    clear_value : Vk::ClearValue
  end

  struct BufferCopy
    src_offset : Vk::DeviceSize
    dst_offset : Vk::DeviceSize
    size : Vk::DeviceSize
  end

  struct BufferCreateInfo
    next : Void*
    size : Vk::DeviceSize
    usage : Vk::BufferUsageFlags
    queue_family_index_count : LibC::UInt
    queue_family_indices : LibC::UInt*
  end

  struct BufferImageCopy
    buffer_offset : Vk::DeviceSize
    buffer_row_length : LibC::UInt
    buffer_image_height : LibC::UInt
    image_subresource : ImageSubresourceLayers
    image_offset : Vk::Offset3D
    image_extent : Vk::Extent3D
  end

  struct BufferViewCreateInfo
    next : Void*
    buffer : Vk::Buffer
    format : Vk::Format
    offset : Vk::DeviceSize
    range : Vk::DeviceSize
  end

  struct ClearAttachment
    color_attachment : LibC::UInt
    clear_value : Vk::ClearValue
  end

  struct ColorBlendAttachmentState
    blend_enable : Vk::Bool32
    src_color_blend_factor : Vk::BlendFactor
    dst_color_blend_factor : Vk::BlendFactor
    color_blend_op : Vk::BlendOp
    src_alpha_blend_factor : Vk::BlendFactor
    dst_alpha_blend_factor : Vk::BlendFactor
    alpha_blend_op : Vk::BlendOp
    color_write_mask : Vk::ColorComponentFlags
  end

  struct ColorBlendState
    next : Void*
    logic_op_enable : Vk::Bool32
    logic_op : Vk::LogicOp
    attachment_count : LibC::UInt
    attachments : ColorBlendAttachmentState*
  end

  struct CommandBufferAllocateInfo
    next : Void*
    queue : Vk::Queue
    command_buffer_count : LibC::UInt
  end

  struct ComputePipelineCreateInfo
    next : Void*
    stage : PipelineShaderStageCreateInfo*
  end

  struct DepthStencilState
    next : Void*
    depth_test_enable : Vk::Bool32
    depth_write_enable : Vk::Bool32
    depth_compare_op : Vk::CompareOp
    depth_bounds_test_enable : Vk::Bool32
    stencil_test_enable : Vk::Bool32
    front : StencilOpState
    back : StencilOpState
  end

  struct DeviceCreateInfo
    next : Void*
    enabled_layer_count : LibC::UInt
    enabled_layer_names : LibC::Char**
    enabled_extension_count : LibC::UInt
    enabled_extension_names : LibC::Char**
  end

  struct FramebufferCreateInfo
    next : Void*
    attachment_count : LibC::UInt
    attachments : Vk::ImageView*
    width : LibC::UInt
    height : LibC::UInt
    layers : LibC::UInt
  end

  struct GraphicsPipelineCreateInfo
    next : Void*
    stage_count : LibC::UInt
    stages : PipelineShaderStageCreateInfo*
  end

  struct ImageBlit
    src_subresource : ImageSubresourceLayers
    src_offsets : Vk::Offset3D[2]
    dst_subresource : ImageSubresourceLayers
    dst_offsets : Vk::Offset3D[2]
  end

  struct ImageCopy
    src_subresource : ImageSubresourceLayers
    src_offset : Vk::Offset3D
    dst_subresource : ImageSubresourceLayers
    dst_offset : Vk::Offset3D
    extent : Vk::Extent3D
  end

  struct ImageCreateInfo
    next : Void*
    flags : Vk::ImageCreateFlags
    image_type : Vk::ImageType
    format : Vk::Format
    extent : Vk::Extent3D
    mip_levels : LibC::UInt
    array_layers : LibC::UInt
    samples : Vk::SampleCountFlagBits
    tiling : Vk::ImageTiling
    usage : Vk::ImageUsageFlags
    queue_family_index_count : LibC::UInt
    queue_family_indices : LibC::UInt*
  end

  struct ImageResolve
    src_subresource : ImageSubresourceLayers
    src_offset : Vk::Offset3D
    dst_subresource : ImageSubresourceLayers
    dst_offset : Vk::Offset3D
    extent : Vk::Extent3D
  end

  struct ImageSubDataInfo
    data_row_length : LibC::UInt
    data_image_height : LibC::UInt
    image_subresource : ImageSubresourceLayers
    image_offset : Vk::Offset3D
    image_extent : Vk::Extent3D
  end

  struct ImageSubresource
    mip_level : LibC::UInt
    array_layer : LibC::UInt
  end

  struct ImageSubresourceLayers
    mip_level : LibC::UInt
    base_array_layer : LibC::UInt
    layer_count : LibC::UInt
  end

  struct ImageSubresourceRange
    base_mip_level : LibC::UInt
    level_count : LibC::UInt
    base_array_layer : LibC::UInt
    layer_count : LibC::UInt
  end

  struct ImageViewCreateInfo
    next : Void*
    image : Vk::Image
    view_type : Vk::ImageViewType
    format : Vk::Format
    components : Vk::ComponentMapping
    subresource_range : ImageSubresourceRange
  end

  struct InputAssemblyState
    next : Void*
    topology : Vk::PrimitiveTopology
    primitive_restart_enable : Vk::Bool32
  end

  struct InstanceCreateInfo
    next : Void*
    application_info : ApplicationInfo*
    enabled_layer_count : LibC::UInt
    enabled_layer_names : LibC::Char**
    enabled_extension_count : LibC::UInt
    enabled_extension_names : LibC::Char**
  end

  struct MappedBufferRange
    buffer : Vk::Buffer
    offset : Vk::DeviceSize
    size : Vk::DeviceSize
  end

  struct MemberInfo
    base_type : BaseType
    offset : LibC::UInt
    size : LibC::UInt
    vec_size : LibC::UInt
    columns : LibC::UInt
    array_size : LibC::UInt
    name : LibC::Char[256]
    next : MemberInfo*
    members : MemberInfo*
  end

  struct MultisampleState
    next : Void*
    rasterization_samples : Vk::SampleCountFlagBits
    sample_shading_enable : Vk::Bool32
    min_sample_shading : LibC::Float
    sample_mask : Vk::SampleMask*
    alpha_to_coverage_enable : Vk::Bool32
    alpha_to_one_enable : Vk::Bool32
  end

  struct PipelineResource
    stages : Vk::ShaderStageFlags
    resource_type : PipelineResourceType
    base_type : BaseType
    access : Vk::AccessFlags
    set : LibC::UInt
    binding : LibC::UInt
    location : LibC::UInt
    input_attachment_index : LibC::UInt
    vec_size : LibC::UInt
    columns : LibC::UInt
    array_size : LibC::UInt
    offset : LibC::UInt
    size : LibC::UInt
    name : LibC::Char[256]
    members : MemberInfo*
  end

  struct PipelineShaderStageCreateInfo
    next : Void*
    module : Vk::ShaderModule
    entry_point : LibC::Char*
    specialization_info : Vk::SpecializationInfo*
  end

  struct PresentInfo
    next : Void*
    wait_semaphore_count : LibC::UInt
    wait_semaphores : Vk::Semaphore*
    wait_dst_stage_mask : Vk::PipelineStageFlags*
    swapchain_count : LibC::UInt
    swapchains : Swapchain*
    images : Vk::Image*
    signal_semaphore_count : LibC::UInt
    signal_semaphores : Vk::Semaphore*
    results : Vk::Result*
  end

  struct QueryPoolCreateInfo
    next : Void*
    query_type : Vk::QueryType
    query_count : LibC::UInt
    pipeline_statistics : Vk::QueryPipelineStatisticFlags
  end

  struct RasterizationState
    next : Void*
    depth_clamp_enable : Vk::Bool32
    rasterizer_discard_enable : Vk::Bool32
    polygon_mode : Vk::PolygonMode
    cull_mode : Vk::CullModeFlags
    front_face : Vk::FrontFace
    depth_bias_enable : Vk::Bool32
  end

  struct RenderPassBeginInfo
    next : Void*
    framebuffer : Framebuffer
    attachment_count : LibC::UInt
    attachments : AttachmentInfo*
  end

  struct SamplerCreateInfo
    next : Void*
    mag_filter : Vk::Filter
    min_filter : Vk::Filter
    mipmap_mode : Vk::SamplerMipmapMode
    address_mode_u : Vk::SamplerAddressMode
    address_mode_v : Vk::SamplerAddressMode
    address_mode_w : Vk::SamplerAddressMode
    mip_lod_bias : LibC::Float
    anisotropy_enable : Vk::Bool32
    max_anisotropy : LibC::Float
    compare_enable : Vk::Bool32
    compare_op : Vk::CompareOp
    min_lod : LibC::Float
    max_lod : LibC::Float
    border_color : Vk::BorderColor
    unnormalized_coordinates : Vk::Bool32
  end

  struct ShaderModuleCreateInfo
    next : Void*
    stage : Vk::ShaderStageFlagBits
    code_size : LibC::SizeT
    code : LibC::UInt*
    glsl_source : LibC::Char*
    entry_point : LibC::Char*
  end

  struct StencilOpState
    fail_op : Vk::StencilOp
    pass_op : Vk::StencilOp
    depth_fail_op : Vk::StencilOp
    compare_op : Vk::CompareOp
  end

  struct SubmitInfo
    next : Void*
    wait_semaphore_count : LibC::UInt
    wait_semaphores : Vk::Semaphore*
    wait_dst_stage_mask : Vk::PipelineStageFlags*
    command_buffer_count : LibC::UInt
    command_buffers : Vk::CommandBuffer*
    signal_semaphore_count : LibC::UInt
    signal_semaphores : Vk::Semaphore*
  end

  struct SubresourceLayout
    offset : Vk::DeviceSize
    size : Vk::DeviceSize
    row_pitch : Vk::DeviceSize
    array_pitch : Vk::DeviceSize
    depth_pitch : Vk::DeviceSize
  end

  struct SwapchainCreateInfo
    next : Void*
    surface : Vk::SurfaceKHR
    format : Vk::SurfaceFormatKHR
    triple_buffer : Vk::Bool32
  end

  struct VertexInputFormatCreateInfo
    vertex_binding_description_count : LibC::UInt
    vertex_binding_descriptions : Vk::VertexInputBindingDescription*
    vertex_attribute_description_count : LibC::UInt
    vertex_attribute_descriptions : Vk::VertexInputAttributeDescription*
  end

  # These were originally types
  alias Framebuffer = Void*
  alias Pipeline = Void*
  alias Swapchain = Void*
  alias VertexInputFormat = Void*
end