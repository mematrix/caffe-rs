use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use static_init::dynamic;
use paste::paste;

use crate::proto::caffe::LayerParameter;
use crate::blob::BlobType;
use crate::layer::{Layer, CaffeLayer, LayerImpl, BlobVec};


pub type LayerCreator<T> = fn(&LayerParameter) -> Rc<RefCell<Layer<T>>>;

#[derive(Default)]
pub struct LayerRegistry<T: BlobType> {
    registry: HashMap<String, LayerCreator<T>>,
}

impl<T: BlobType> LayerRegistry<T> {
    pub fn new() -> Self {
        Default::default()
    }

    /// Add a creator.
    pub fn add_creator(&mut self, ty: &str, creator: LayerCreator<T>) {
        assert!(!self.registry.contains_key(ty), "Layer type {:?} already registered.", ty);

        self.registry.insert(ty.to_string(), creator);
    }

    pub fn create_layer(&self, param: &LayerParameter) -> Rc<RefCell<Layer<T>>> {
        // todo: add caffe::root_solver()

        let ty = param.get_field_type();
        match self.registry.get(ty) {
            Some(creator) => creator(param),
            None => panic!("Unknown layer type: {:?} (known types: {:?})", ty, self.layer_type_list_string()),
        }
    }

    pub fn layer_type_list(&self) -> Vec<String> {
        let mut layer_types = Vec::with_capacity(self.registry.len());
        for (k, _) in &self.registry {
            layer_types.push(k.clone());
        }

        layer_types
    }

    fn layer_type_list_string(&self) -> String {
        self.layer_type_list().join(" ,")
    }
}


#[dynamic]
static mut REGISTRY_F32: LayerRegistry<f32> = LayerRegistry::new();

#[dynamic]
static mut REGISTRY_F64: LayerRegistry<f64> = LayerRegistry::new();

pub trait LayerRegister<T: BlobType> {
    fn register(ty: &str, creator: LayerCreator<T>);
}

pub struct LayerRegisterImpl<T: BlobType> {
    phantom: PhantomData<T>,
}

impl<T: BlobType> LayerRegister<T> for LayerRegisterImpl<T> {
    default fn register(_ty: &str, _creator: LayerCreator<T>) {
        unimplemented!();
    }
}

impl LayerRegister<f32> for LayerRegisterImpl<f32> {
    fn register(ty: &str, creator: LayerCreator<f32>) {
        let mut lock = REGISTRY_F32.write();
        lock.add_creator(ty, creator);
    }
}

impl LayerRegister<f64> for LayerRegisterImpl<f64> {
    fn register(ty: &str, creator: LayerCreator<f64>) {
        let mut lock = REGISTRY_F64.write();
        lock.add_creator(ty, creator);
    }
}

impl<T: BlobType> LayerRegisterImpl<T> {
    pub fn new(ty: &str, creator: LayerCreator<T>) -> Self {
        LayerRegisterImpl::<T>::register(ty, creator);
        LayerRegisterImpl {
            phantom: PhantomData
        }
    }
}


macro_rules! register_layer_creator {
    ($t:ident, $creator:path) => {
        paste! {
            #[dynamic(init)]
            static [<G_CREATOR_F_ $t:snake:upper>]: $crate::layer_factory::LayerRegisterImpl<f32> =
                $crate::layer_factory::LayerRegisterImpl::<f32>::new(stringify!($t), $creator);

            #[dynamic(init)]
            static [<G_CREATOR_D_ $t:snake:upper>]: $crate::layer_factory::LayerRegisterImpl<f64> =
                $crate::layer_factory::LayerRegisterImpl::<f64>::new(stringify!($t), $creator);
        }
    };
}

macro_rules! register_layer_class {
    ($t:ident) => {
        paste! {
            pub fn [<create_ $t:snake layer>]<T: 'static + $crate::blob::BlobType>(param: &crate::proto::caffe::LayerParameter)
                -> std::rc::Rc<std::cell::RefCell<$crate::layer::Layer<T>>> {
                std::rc::Rc::new(std::cell::RefCell::new($crate::layer::Layer::new(Box::new($t::<T>::new(param)))))
            }

            register_layer_creator!($t, self::[<create_ $t:snake layer>]);
        }
    };
}


#[cfg(test)]
mod test {
    use super::*;

    fn test<T: BlobType>(p: &LayerParameter) -> Rc<RefCell<Layer<T>>> {
        unimplemented!();
    }

    static TTT: LayerCreator<f32> = test::<f32>;

    register_layer_creator!(PhantomData, self::test);

    struct TestLayer<T: BlobType> {
        phantom: PhantomData<T>,
    }

    impl<T: BlobType> CaffeLayer<T> for TestLayer<T> {
        fn get_impl(&self) -> &LayerImpl<T> {
            todo!()
        }

        fn get_impl_mut(&mut self) -> &mut LayerImpl<T> {
            todo!()
        }

        fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
            todo!()
        }

        fn forward_cpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
            todo!()
        }

        fn backward_cpu(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
            todo!()
        }
    }

    impl<T: BlobType> TestLayer<T> {
        pub fn new(param: &LayerParameter) -> Self {
            TestLayer {
                phantom: PhantomData
            }
        }
    }

    register_layer_class!(TestLayer);
}
