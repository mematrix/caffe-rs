use std::any::{TypeId, Any};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use static_init::dynamic;
use paste::paste;

use crate::proto::caffe::LayerParameter;
use crate::blob::BlobType;
use crate::layer::{Layer, CaffeLayer, LayerImpl, BlobVec, SharedLayer};


pub type LayerCreator<T> = fn(&LayerParameter) -> SharedLayer<T>;

#[derive(Default)]
struct CreatorRegistry<T: BlobType> {
    registry: HashMap<String, LayerCreator<T>>,
}

impl<T: BlobType> CreatorRegistry<T> {
    pub fn new() -> Self {
        Default::default()
    }

    /// Add a creator.
    pub fn add_creator(&mut self, ty: &str, creator: LayerCreator<T>) {
        assert!(!self.registry.contains_key(ty), "Layer type {:?} already registered.", ty);

        self.registry.insert(ty.to_string(), creator);
    }

    /// Get a layer using a `LayerParameter`.
    pub fn create_layer(&self, param: &LayerParameter) -> SharedLayer<T> {
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

fn add_creator_impl<T: BlobType>(ty: &str, creator: LayerCreator<T>) {
    let mut lock = REGISTRY.write();
    let registry = lock.entry(TypeId::of::<T>()).or_insert_with(|| Box::new(CreatorRegistry::<T>::new()));
    let registry = registry.downcast_mut::<CreatorRegistry<T>>().unwrap();
    registry.add_creator(ty, creator);
}

fn create_layer_impl<T: BlobType>(param: &LayerParameter) -> SharedLayer<T> {
    let mut lock = REGISTRY.write();
    let registry = lock.entry(TypeId::of::<T>()).or_insert_with(|| Box::new(CreatorRegistry::<T>::new()));
    let registry = registry.downcast_ref::<CreatorRegistry<T>>().unwrap();
    registry.create_layer(param)
}

#[dynamic]
static mut REGISTRY: HashMap<TypeId, Box<dyn Any + Send + Sync>> = HashMap::new();

pub struct LayerRegistry<T: BlobType> {
    phantom: PhantomData<T>,
}

impl<T: BlobType> LayerRegistry<T> {
    pub fn new(ty: &str, creator: LayerCreator<T>) -> Self {
        Self::add_creator(ty, creator);
        LayerRegistry {
            phantom: PhantomData
        }
    }

    pub fn add_creator(ty: &str, creator: LayerCreator<T>) {
        add_creator_impl(ty, creator);
    }

    pub fn create_layer(param: &LayerParameter) -> SharedLayer<T> {
        create_layer_impl(param)
    }
}


#[macro_export]
macro_rules! register_layer_creator {
    ($t:ident, $creator:path) => {
        paste! {
            #[dynamic(init)]
            static [<G_CREATOR_F_ $t:snake:upper>]: $crate::layer_factory::LayerRegistry<f32> =
                $crate::layer_factory::LayerRegistry::<f32>::new(stringify!($t), $creator);

            #[dynamic(init)]
            static [<G_CREATOR_D_ $t:snake:upper>]: $crate::layer_factory::LayerRegistry<f64> =
                $crate::layer_factory::LayerRegistry::<f64>::new(stringify!($t), $creator);
        }
    };
}

#[macro_export]
macro_rules! register_layer_class {
    ($t:ident) => {
        paste! {
            pub fn [<create_ $t:snake _layer>]<T: 'static + $crate::blob::BlobType>(param: &crate::proto::caffe::LayerParameter)
                -> std::rc::Rc<std::cell::RefCell<$crate::layer::Layer<T>>> {
                std::rc::Rc::new(std::cell::RefCell::new($crate::layer::Layer::new(Box::new([<$t Layer>]::<T>::new(param)))))
            }

            register_layer_creator!($t, self::[<create_ $t:snake _layer>]);
        }
    };
}


#[cfg(test)]
mod test {
    use super::*;

    fn test<T: BlobType>(p: &LayerParameter) -> SharedLayer<T> {
        unimplemented!();
    }

    static TTT: LayerCreator<f32> = test::<f32>;

    register_layer_creator!(PhantomData, self::test);

    struct TestLayer<T: BlobType> {
        phantom: PhantomData<T>,
    }

    impl<T: BlobType> CaffeLayer for TestLayer<T> {
        type DataType = T;

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
        pub fn new(_param: &LayerParameter) -> Self {
            TestLayer {
                phantom: PhantomData
            }
        }
    }

    register_layer_class!(Test);
}
