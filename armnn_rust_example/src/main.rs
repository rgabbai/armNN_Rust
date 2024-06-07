extern crate libc;

use std::ffi::CString;
use image::{self, GenericImageView, FilterType};
use ndarray::Array;
use ndarray::IxDyn;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));



// Returns the input tensor, original image width and height
fn prepare_input<P: AsRef<Path>>(file_path: P) -> (Array<f32, IxDyn>, u32, u32) {
    // Load the image from the file
    let img = image::open(file_path).unwrap();
    
    let (img_width, img_height) = (img.width(), img.height());
    
    // Resize the image to 640x640
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);
    
    // Create an input array with dimensions (1, 3, 640, 640)
    let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
    
    // Fill the input array with the image data
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.0;
        input[[0, 1, y, x]] = (g as f32) / 255.0;
        input[[0, 2, y, x]] = (b as f32) / 255.0;
    }
    
    (input, img_width, img_height)
}


// Function used to convert RAW output from YOLOv8 to an array
// of detected objects. Each object contain the bounding box of
// this object, the type of object and the probability
// Returns array of detected objects in a format [(x1,y1,x2,y2,object_type,probability),..]
fn process_output(output:Array<f32,IxDyn>,img_width: u32, img_height: u32, model:&str, thr:f32) -> Vec<(f32,f32,f32,f32,&'static str, f32)> {

    let mut yolo_class = [" "," "," "];
    match model {
        "A" => {
            yolo_class = ["hen", "bucket", "cone"];
        },
        "B" => {
            yolo_class = ["pylon", "person", "roktrack"];
        },
        _ => unreachable!("Mode should be either 'A' or 'B'"), // This case should never happen 
    }
    let mut boxes = Vec::new();
    let output = output.slice(s![..,..,0]);
    for row in output.axis_iter(Axis(0)) {
        
        let row:Vec<_> = row.iter().map(|x| *x).collect();
        //find the index with higest probability of of the 80 classes.
        let (class_id, prob) = row.iter().skip(4).enumerate()
            .map(|(index,value)| (index,*value))
            .reduce(|accum, row| if row.1>accum.1 { row } else {accum}).unwrap(); 
        if prob < thr {
            continue
        }
        //println!("Row: {:?}",row);
        //println!("Class:{class_id}:{prob}");
        let label = yolo_class[class_id];
        let xc = row[0]/640.0*(img_width as f32);
        let yc = row[1]/640.0*(img_height as f32);
        let w = row[2]/640.0*(img_width as f32);
        let h = row[3]/640.0*(img_height as f32);
        let x1 = xc - w/2.0;
        let x2 = xc + w/2.0;
        let y1 = yc - h/2.0;
        let y2 = yc + h/2.0;

        let prob = round_to_decimal_places(prob,1);
        boxes.push((x1,y1,x2,y2,label,prob));
    }
    //println!("Boxes:{:?}",boxes);
    boxes.sort_by(|box1,box2| box2.5.total_cmp(&box1.5));
    //println!("Ordered Boxes:{:?}",boxes);
    let mut result = Vec::new();
    // Remove duplicated detections - assume hieghest probability is taken in each class
    // TBD - why the classes are not mixed after we sort with probability 
    while boxes.len()>0 {
        result.push(boxes[0]);
        //println!("Box[0]:{:?}",boxes[0]);
        //println!("Boxes:{:?}",boxes);
        boxes = boxes.iter().filter(|box1| iou(&boxes[0],box1) < 0.7).map(|x| *x).collect()
    }
    return result;
}


fn main() {
    // Initialize Arm NN runtime
    let runtime = unsafe { armnn::IRuntime::Create(armnn::IRuntime::CreationOptions {}) };
    if runtime.is_null() {
        eprintln!("Failed to create Arm NN runtime.");
        return;
    }

    // Load the model
    let model_path = CString::new("/src/yolov8n_hen_bucket_cone_640.tflite").expect("CString::new failed");
    let parser = unsafe { armnnTfLiteParser::ITfLiteParser::Create() };
    let network = unsafe {
        (*parser).CreateNetworkFromBinaryFile(
            model_path.as_ptr(),
            armnn::Compute::CpuAcc,
        )
    };

    // Optimize the network
    let optimized_network = unsafe {
        armnn::Optimize(
            *network,
            &armnn::Compute::CpuAcc as *const _,
            runtime,
        )
    };

    // Load the optimized network into the runtime
    let mut network_id: armnn::NetworkId = 0;
    let status = unsafe {
        (*runtime).LoadNetwork(&mut network_id, optimized_network, &armnn::NullBackendOptions {})
    };
    if status != armnn::Status::Success {
        eprintln!("Failed to load the network into the runtime.");
        return;
    }

    // Prepare input and output tensors
   
   
    //let input_tensor_data: Vec<f32> = vec![0.0;640 * 640 * 3]; // Fill with actual data
    let file_path = "/src/test_image1.jpg";
    let (input_tensor_data, width, height) = prepare_input(file_path);
   
   
    let mut output_tensor_data: Vec<f32> = vec![0.0; 1000];

    let input_tensors = armnn::InputTensors {
        0: armnn::ConstTensor {
            binding_info: armnn::BindingInfo {
                binding_id: 0,
                tensor_info: (*runtime).GetInputTensorInfo(network_id, 0),
            },
            data: input_tensor_data.as_ptr() as *const libc::c_void,
        },
    };

    let output_tensors = armnn::OutputTensors {
        0: armnn::Tensor {
            binding_info: armnn::BindingInfo {
                binding_id: 0,
                tensor_info: (*runtime).GetOutputTensorInfo(network_id, 0),
            },
            data: output_tensor_data.as_mut_ptr() as *mut libc::c_void,
        },
    };

    // Execute the inference
    let status = unsafe { (*runtime).EnqueueWorkload(network_id, &input_tensors, &output_tensors) };
    if status != armnn::Status::Success {
        eprintln!("Failed to execute the inference.");
        return;
    }
    println!("Run NN done!");
    // Process the output data
    // ... (process output_tensor_data)
}
