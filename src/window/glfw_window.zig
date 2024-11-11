const GlfwWindow = @This();
const glfw = @import("../c.zig");

var window: *glfw.GLFWwindow = undefined;

pub fn init(width: u32, height: u32, title: [:0]const u8) !void {
    if (glfw.glfwInit() != glfw.GLFW_TRUE) return error.WindowCreationFailed;
    glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_NO_API);
    // const result = glfw.glfwInit();
    // assert(result != glfw.GLFW_TRUE);

    const glfw_window = glfw.glfwCreateWindow(
        @intCast(width),
        @intCast(height),
        title,
        null,
        null,
    ) orelse return error.WindowCreationFailed;

    window = glfw_window;
}

pub fn deinit() void {
    glfw.glfwDestroyWindow(window);
    glfw.glfwTerminate();
}

pub fn getWindow() *glfw.GLFWwindow {
    return window;
}

pub fn windowClosed() bool {
    return glfw.glfwWindowShouldClose(window) == glfw.GLFW_TRUE;
}

pub fn update() void {
    glfw.glfwPollEvents();
}
