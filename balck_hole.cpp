#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>

// --- Dear ImGui Includes (Assuming external setup/library files are present) ---
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace glm;
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// --- GLOBAL CONSTANTS & VARS ---
const double TIME_STEP = 0.01; // Fixed step for physics integration (seconds)
double lastPrintTime = 0.0;
int    framesCount   = 0;
double c = 299792458.0;
double G = 6.67430e-11;
struct Ray;
bool Gravity = false;

struct Camera {
    // Center the camera orbit on the black hole at (0, 0, 0)
    vec3 target = vec3(0.0f, 0.0f, 0.0f); // Always look at the black hole center
    float radius = 6.34194e10f;
    float minRadius = 1e10f, maxRadius = 1e12f;

    float azimuth = 0.0f;
    float elevation = M_PI / 2.0f;

    float orbitSpeed = 0.01f;
    float panSpeed = 0.01f;
    double zoomSpeed = 25e9f;

    bool dragging = false;
    bool panning = false;
    bool moving = false; // For compute shader optimization
    double lastX = 0.0, lastY = 0.0;

    // Calculate camera position in world space
    vec3 position() const {
        float clampedElevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        // Orbit around (0,0,0) always
        return vec3(
            radius * sin(clampedElevation) * cos(azimuth),
            radius * cos(clampedElevation),
            radius * sin(clampedElevation) * sin(azimuth)
        );
    }
    void update() {
        // Always keep target at black hole center
        target = vec3(0.0f, 0.0f, 0.0f);
        if(dragging | panning) {
            moving = true;
        } else {
            moving = false;
        }
    }

    void processMouseMove(double x, double y) {
        float dx = float(x - lastX);
        float dy = float(y - lastY);

        if (dragging && panning) {
            // Pan: Shift + Left or Middle Mouse - disabled to keep focus on BH
        }
        else if (dragging && !panning) {
            // Orbit: Left mouse only
            azimuth   += dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        }

        lastX = x;
        lastY = y;
        update();
    }
    void processMouseButton(int button, int action, int mods, GLFWwindow* win) {
        // Let ImGui handle input first
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) return;

        if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                dragging = true;
                panning = false; // Disabled
                glfwGetCursorPos(win, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                dragging = false;
                panning = false;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            // Right click currently toggles Gravity in the original code, but we'll use the UI now.
        }
    }
    void processScroll(double xoffset, double yoffset) {
        // Let ImGui handle input first
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) return;

        radius -= yoffset * zoomSpeed;
        radius = glm::clamp(radius, minRadius, maxRadius);
        update();
    }
    void processKey(int key, int scancode, int action, int mods) {
        // Let ImGui handle input first
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) return;

        if (action == GLFW_PRESS && key == GLFW_KEY_G) {
            Gravity = !Gravity;
            cout << "[INFO] Gravity turned " << (Gravity ? "ON" : "OFF") << endl;
        }
    }
};
Camera camera;

struct BlackHole {
    vec3 position;
    double mass;
    double radius;
    double r_s;

    BlackHole(vec3 pos, double m) : position(pos), mass(m) {r_s = 2.0 * G * mass / (c*c);}
    bool Intercept(float px, float py, float pz) const {
        double dx = double(px) - double(position.x);
        double dy = double(py) - double(position.y);
        double dz = double(pz) - double(position.z);
        double dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < r_s * r_s;
    }
    void updateRadius() {
        r_s = 2.0 * G * mass / (c*c);
    }
};
// Initializing SagA with double mass for precision
BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36); // Sagittarius A black hole

struct ObjectData {
    vec4 posRadius; // xyz = position, w = radius
    vec4 color;     // rgb = color, a = unused
    double  mass;  // Use double for physics calculation
    vec3 velocity = vec3(0.0f, 0.0f, 0.0f); // Initial velocity
};

// Initial objects setup (first two are stars, third is the black hole itself)
vector<ObjectData> objects = {
    { vec4(4e11f, 0.0f, 0.0f, 4e9f)   , vec4(1,1,0,1), 1.98892e30 },
    { vec4(0.0f, 0.0f, 4e11f, 4e9f)   , vec4(1,0,0,1), 1.98892e30 },
    { vec4(0.0f, 0.0f, 0.0f, static_cast<float>(SagA.r_s)) , vec4(0,0,0,1), SagA.mass },
};

struct Engine {
    GLuint gridShaderProgram;
    // -- Quad & Texture render -- //
    GLFWwindow* window;
    GLuint quadVAO;
    GLuint texture;
    GLuint shaderProgram;
    GLuint computeProgram = 0;
    // -- UBOs -- //
    GLuint cameraUBO = 0;
    GLuint diskUBO = 0;
    GLuint objectsUBO = 0;
    // -- grid mess vars -- //
    GLuint gridVAO = 0;
    GLuint gridVBO = 0;
    GLuint gridEBO = 0;
    int gridIndexCount = 0;

    int WIDTH = 1280;  // Window width
    int HEIGHT = 720; // Window height
    int COMPUTE_WIDTH  = 320;   // Compute resolution width
    int COMPUTE_HEIGHT = 180;  // Compute resolution height
    float width = 100000000000.0f; // Width of the viewport in meters
    float height = 75000000000.0f; // Height of the viewport in meters
    
    Engine() {
        // 1. GLFW/GLEW Setup (as in original code)
        if (!glfwInit()) {
            cerr << "GLFW init failed\n";
            exit(EXIT_FAILURE);
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole TDA Dynamics", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        GLenum glewErr = glewInit();
        if (glewErr != GLEW_OK) {
            cerr << "Failed to initialize GLEW: " << (const char*)glewGetErrorString(glewErr) << "\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        cout << "OpenGL " << glGetString(GL_VERSION) << "\n";

        // 2. ImGui Setup
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 430");

        // 3. Shader and UBO Setup (as in original code)
        this->shaderProgram = CreateShaderProgram();
        gridShaderProgram = CreateShaderProgram("grid.vert", "grid.frag");
        computeProgram = CreateComputeProgram("geodesic.comp");

        // Camera UBO (binding 1)
        glGenBuffers(1, &cameraUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW); 
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO); 

        // Disk UBO (binding 2)
        glGenBuffers(1, &diskUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO); 

        // Objects UBO (binding 3)
        glGenBuffers(1, &objectsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        GLsizeiptr objUBOSize = sizeof(int) + 3 * sizeof(float)
            + 16 * (sizeof(vec4) + sizeof(vec4))
            + 16 * sizeof(float); // 16 floats for mass (use float for GLSL transfer)
        glBufferData(GL_UNIFORM_BUFFER, objUBOSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO); 

        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
        
        // Setup window resize callback
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int w, int h) {
            Engine* eng = (Engine*)glfwGetWindowUserPointer(win);
            eng->WIDTH = w;
            eng->HEIGHT = h;
            glViewport(0, 0, w, h);
        });
    }

    ~Engine() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void generateGrid(const vector<ObjectData>& objects) {
        const int gridSize = 25;
        const float spacing = 1e10f;  // tweak this

        vector<vec3> vertices;
        vector<GLuint> indices;

        for (int z = 0; z <= gridSize; ++z) {
            for (int x = 0; x <= gridSize; ++x) {
                float worldX = (x - gridSize / 2) * spacing;
                float worldZ = (z - gridSize / 2) * spacing;

                float y = 0.0f;

                // ✅ Warp grid using Schwarzschild geometry
                for (const auto& obj : objects) {
                    // Skip the black hole itself as its influence is captured in SagA
                    if (obj.posRadius.w == SagA.r_s) continue; 
                    
                    vec3 objPos = vec3(obj.posRadius);
                    double mass = obj.mass;
                    double r_s_obj = 2.0 * G * mass / (c * c); // Schwarzschild radius for this object

                    double dx = worldX - objPos.x;
                    double dz = worldZ - objPos.z;
                    double dist = sqrt(dx * dx + dz * dz);

                    // Apply the Schwarzschild 'warp' effect
                    if (dist > r_s_obj) {
                        // Formula for embedding diagram
                        double deltaY = 2.0 * sqrt(r_s_obj * (dist - r_s_obj));
                        y -= static_cast<float>(deltaY) * 5e-1f; // Adjust scale for visibility
                    } else if (dist > 1e-10) { // Near or inside the horizon
                        y -= static_cast<float>(2.0 * r_s_obj);
                    }
                }
                
                vertices.emplace_back(worldX, y, worldZ);
            }
        }

        // 🧩 Add indices for GL_LINE rendering
        int numVertices = gridSize + 1;
        for (int z = 0; z < gridSize; ++z) {
            for (int x = 0; x < gridSize; ++x) {
                int i = z * numVertices + x;
                // Horizontal line segment
                indices.push_back(i);
                indices.push_back(i + 1);
                // Vertical line segment
                indices.push_back(i);
                indices.push_back(i + numVertices);
            }
            // Last column vertical line
            indices.push_back((z + 1) * numVertices - 1);
            indices.push_back((z + 2) * numVertices - 1);
        }
        // Last row horizontal line
        for (int x = 0; x < gridSize; ++x) {
            indices.push_back(gridSize * numVertices + x);
            indices.push_back(gridSize * numVertices + x + 1);
        }

        // 🔌 Upload to GPU
        if (gridVAO == 0) glGenVertexArrays(1, &gridVAO);
        if (gridVBO == 0) glGenBuffers(1, &gridVBO);
        if (gridEBO == 0) glGenBuffers(1, &gridEBO);

        glBindVertexArray(gridVAO);

        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // location = 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);

        gridIndexCount = indices.size();

        glBindVertexArray(0);
    }
    void drawGrid(const mat4& viewProj) {
        glUseProgram(gridShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram, "viewProj"),
                        1, GL_FALSE, glm::value_ptr(viewProj));
        glBindVertexArray(gridVAO);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(1.0f); // Make grid lines thin

        glDrawElements(GL_LINES, gridIndexCount, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
    }
    void drawFullScreenQuad() {
        glUseProgram(shaderProgram); // fragment + vertex shader
        glBindVertexArray(quadVAO);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

        glDisable(GL_DEPTH_TEST);  // draw as background
        glDrawArrays(GL_TRIANGLES, 0, 6);  // 2 triangles
        glEnable(GL_DEPTH_TEST);
    }
    GLuint CreateShaderProgram(){
        // Full-screen quad drawing shader (hardcoded)
        const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        })";

        const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        })";

        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);

        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return shaderProgram;
    };

    GLuint loadAndCompileShader(const char* path, GLenum type) {
        std::ifstream in(path);
        if (!in.is_open()) {
            std::cerr << "Failed to open shader: " << path << "\n";
            exit(EXIT_FAILURE);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        std::string srcStr = ss.str();
        const char* src = srcStr.c_str();

        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLint logLen;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetShaderInfoLog(shader, logLen, nullptr, log.data());
            std::cerr << "Shader compile error (" << path << "):\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }
        return shader;
    }

    GLuint CreateShaderProgram(const char* vertPath, const char* fragPath) {
        GLuint vertShader = loadAndCompileShader(vertPath, GL_VERTEX_SHADER);
        GLuint fragShader = loadAndCompileShader(fragPath, GL_FRAGMENT_SHADER);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);

        GLint linkSuccess;
        glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
        if (!linkSuccess) {
            GLint logLen;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(program, logLen, nullptr, log.data());
            std::cerr << "Shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(vertShader);
        glDeleteShader(fragShader);

        return program;
    }

    GLuint CreateComputeProgram(const char* path) {
        GLuint cs = loadAndCompileShader(path, GL_COMPUTE_SHADER);
        
        // 3) link
        GLuint prog = glCreateProgram();
        glAttachShader(prog, cs);
        glLinkProgram(prog);
        
        GLint ok;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(prog, logLen, nullptr, log.data());
            std::cerr << "Compute shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(cs);
        return prog;
    }
    
    void dispatchCompute(const Camera& cam) {
        // determine target compute‐res
        // If camera is moving, use lower resolution for performance
        int cw = cam.moving ? COMPUTE_WIDTH  : WIDTH; 
        int ch = cam.moving ? COMPUTE_HEIGHT : HEIGHT;

        // 1) reallocate the texture if needed
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cw, ch, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // 2) bind compute program & UBOs
        glUseProgram(computeProgram);
        uploadCameraUBO(cam);
        uploadDiskUBO();
        uploadObjectsUBO(objects);

        // Set compute resolution uniforms
        glUniform2f(glGetUniformLocation(computeProgram, "iResolution"), (float)cw, (float)ch);

        // 3) bind it as image unit 0
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        // 4) dispatch grid
        GLuint groupsX = (GLuint)std::ceil(cw / 16.0f);
        GLuint groupsY = (GLuint)std::ceil(ch / 16.0f);
        glDispatchCompute(groupsX, groupsY, 1);

        // 5) sync
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    
    void uploadCameraUBO(const Camera& cam) {
        // Ensure a consistent structure for GLSL std140 layout
        struct UBOData {
            vec3 pos; float _pad0;
            vec3 right; float _pad1;
            vec3 up; float _pad2;
            vec3 forward; float _pad3;
            float tanHalfFov;
            float aspect;
            int moving; // Use int for bool to ensure std140 compliance
            int _pad4;
        } data;
        vec3 fwd = normalize(cam.target - cam.position());
        vec3 up = vec3(0, 1, 0); // y axis is up, so disk is in x-z plane
        vec3 right = normalize(cross(fwd, up));
        up = cross(right, fwd);

        data.pos = cam.position();
        data.right = right;
        data.up = up;
        data.forward = fwd;
        data.tanHalfFov = tan(radians(60.0f * 0.5f));
        data.aspect = float(WIDTH) / float(HEIGHT);
        data.moving = cam.dragging || cam.panning;

        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UBOData), &data);
    }
    void uploadObjectsUBO(const vector<ObjectData>& objs) {
        // Mass is a float array in the UBO structure to match GLSL std140 layout
        struct UBOData {
            int   numObjects;
            float _pad0, _pad1, _pad2;
            vec4  posRadius[16];
            vec4  color[16];
            float  mass[16];
        } data;

        size_t count = std::min(objs.size(), size_t(16));
        data.numObjects = static_cast<int>(count);

        for (size_t i = 0; i < count; ++i) {
            data.posRadius[i] = objs[i].posRadius;
            data.color[i] = objs[i].color;
            data.mass[i] = static_cast<float>(objs[i].mass); // Cast double mass to float for GLSL
        }

        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
    }
    void uploadDiskUBO() {
        // r_s is updated by the BlackHole struct whenever its mass changes
        float r1 = SagA.r_s * 2.2f;
        float r2 = SagA.r_s * 5.2f;
        float num = 2.0;
        float thickness = 1e9f;
        float diskData[4] = { r1, r2, num, thickness };

        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(diskData), diskData);
    }
    
    vector<GLuint> QuadVAO(){
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
            1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
            1.0f, -1.0f,  1.0f, 0.0f,
            1.0f,  1.0f,  1.0f, 1.0f
        };
        
        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // Initial texture allocation (will be reallocated on compute dispatch)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        vector<GLuint> VAOtexture = {VAO, texture};
        return VAOtexture;
    }
};
Engine engine;

// TDA Dynamics UI for ImGui
void drawTdaDynamicsUI() {
    ImGui::Begin("TDA Dynamics Controls");

    ImGui::Text("Simulation Time Step: %.2f s/frame", TIME_STEP);
    ImGui::Separator();

    // --- General Dynamics Control ---
    ImGui::Text("General Dynamics");
    ImGui::Checkbox("N-Body Gravity Simulation", &Gravity);
    ImGui::SameLine();
    ImGui::TextDisabled("(G to toggle)");

    ImGui::Text("Black Hole (Sagittarius A*)");
    // Input double for BH mass. Use a local float array to interface with ImGui.
    double bh_mass = SagA.mass;
    if (ImGui::InputDouble("BH Mass (kg)", &bh_mass, 1e36, 10e36, "%.2e")) {
        SagA.mass = glm::clamp(bh_mass, 1e36, 1e38);
        SagA.updateRadius();
        // Update the BH object in the main list
        objects[2].mass = SagA.mass;
        objects[2].posRadius.w = static_cast<float>(SagA.r_s);
    }
    ImGui::Text("Schwarzschild Radius: %.2e m", SagA.r_s);
    ImGui::Separator();

    // --- Object Controls ---
    ImGui::Text("TDA Objects (Stars/Masses)");
    // Start from index 0, skip the last object which is the black hole (index 2)
    for (size_t i = 0; i < 2; ++i) {
        ImGui::PushID(i);
        ObjectData& obj = objects[i];

        ImGui::Text("Object %zu", i + 1);
        
        // Position
        ImGui::InputFloat3("Position (m)", glm::value_ptr(obj.posRadius), "%.2e");
        
        // Velocity
        if (ImGui::InputFloat3("Velocity (m/s)", glm::value_ptr(obj.velocity), "%.2f")) {
            camera.moving = true;
        }

        // Mass
        double mass_val = obj.mass;
        if (ImGui::InputDouble("Mass (kg)", &mass_val, 1e29, 1e31, "%.2e")) {
            obj.mass = glm::clamp(mass_val, 1e29, 1e35);
        }

        // Radius (visual size)
        if (ImGui::InputFloat("Radius (m)", &obj.posRadius.w, 1e8, 1e9, "%.1e")) {
            obj.posRadius.w = glm::clamp(obj.posRadius.w, 1e8f, 1e11f);
        }

        // Color
        ImGui::ColorEdit3("Color", glm::value_ptr(obj.color));

        ImGui::Separator();
        ImGui::PopID();
    }
    
    // --- Camera Controls ---
    ImGui::Text("Camera Controls");
    ImGui::SliderFloat("Orbit Speed", &camera.orbitSpeed, 0.001f, 0.1f, "%.3f");
    ImGui::InputDouble("Zoom Speed (m/scroll)", &camera.zoomSpeed, 1e9, 1e10, "%.2e");

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::End();
}


void setupCameraCallbacks(GLFWwindow* window) {
    // Note: ImGui handlers are set up in Engine constructor via ImGui_ImplGlfw_InitForOpenGL
    // The custom callbacks below manually check ImGui::GetIO().WantCapture...
    
    glfwSetWindowUserPointer(window, &camera);

    // Use ImGui's built-in mouse button callback, but also set up custom logic
    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        ImGui_ImplGlfw_MouseButtonCallback(win, button, action, mods); // Pass to ImGui first
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseButton(button, action, mods, win);
    });

    // Use ImGui's built-in cursor position callback, but also set up custom logic
    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
        ImGui_ImplGlfw_CursorPosCallback(win, x, y); // Pass to ImGui first
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseMove(x, y);
    });

    // Use ImGui's built-in scroll callback, but also set up custom logic
    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoffset, double yoffset) {
        ImGui_ImplGlfw_ScrollCallback(win, xoffset, yoffset); // Pass to ImGui first
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processScroll(xoffset, yoffset);
    });

    // Use ImGui's built-in key callback, but also set up custom logic
    glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
        ImGui_ImplGlfw_KeyCallback(win, key, scancode, action, mods); // Pass to ImGui first
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processKey(key, scancode, action, mods);
    });
}


// -- MAIN -- //
int main() {
    // Initialize the engine which sets up GLFW, GLEW, ImGui, and Shaders
    // Engine engine; // Already globally declared, use this if not global: Engine engine;
    setupCameraCallbacks(engine.window);

    // Main loop
    while (!glfwWindowShouldClose(engine.window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // --- 1. Physics Update (TDA Dynamics) ---
        if (Gravity) {
            // Only update non-BH objects (index < 2). BH is static at index 2.
            for (size_t i = 0; i < 2; ++i) {
                ObjectData& obj = objects[i];
                vec3 acceleration = vec3(0.0f);

                // Sum gravity forces from all other objects, including the BH (index 2)
                for (size_t j = 0; j < objects.size(); ++j) {
                    if (i == j) continue;

                    const ObjectData& obj2 = objects[j];
                    
                    vec3 pos1 = vec3(obj.posRadius);
                    vec3 pos2 = vec3(obj2.posRadius);
                    
                    vec3 direction = pos2 - pos1;
                    float distance = length(direction);
                    
                    if (distance > 1e-9f) {
                        // F = G * m1 * m2 / r^2
                        // a = F / m1 = G * m2 / r^2
                        double acc_mag = (G * obj2.mass) / (distance * distance);
                        
                        // Calculate acceleration vector
                        acceleration += (direction / distance) * (float)acc_mag;
                    }
                }
                
                // Simple Euler Integration (Velocity Verlet or RK4 would be better)
                obj.velocity += acceleration * (float)TIME_STEP;
                
                // Update position
                obj.posRadius.x += obj.velocity.x * (float)TIME_STEP;
                obj.posRadius.y += obj.velocity.y * (float)TIME_STEP;
                obj.posRadius.z += obj.velocity.z * (float)TIME_STEP;

                // Mark camera as moving to trigger higher resolution raytracing
                camera.moving = true; 
            }
        }

        // --- 2. Grid (Spacetime Bending) ---
        engine.generateGrid(objects);
        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f), (float)engine.WIDTH/engine.HEIGHT, 1e9f, 1e14f);
        mat4 viewProj = proj * view;
        engine.drawGrid(viewProj);

        // --- 3. Raytracing (Compute Shader) ---
        glViewport(0, 0, engine.WIDTH, engine.HEIGHT);
        engine.dispatchCompute(camera);
        engine.drawFullScreenQuad();

        // --- 4. ImGui UI Rendering ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        drawTdaDynamicsUI(); // The new TDA Dynamics control panel
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // --- 5. Present ---
        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}
