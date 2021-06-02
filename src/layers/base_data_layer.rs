use crate::blob::{BlobType, Blob, ArcBlob};
use crate::data_transformer::DataTransformer;
use crate::internal_thread::{InternalThread, CancelToken, InternalThreadImpl};
use crate::layer::{LayerImpl, BlobVec};
use crate::proto::caffe::{TransformationParameter, LayerParameter};
use crate::util::blocking_queue::BlockingQueue;


/// Provides base for data layers that feed blobs to the Net.
pub struct BaseDataLayerImpl<T: BlobType> {
    pub layer: LayerImpl<T>,
    pub transform_param: TransformationParameter,
    pub data_transformer: Option<DataTransformer<T>>,
    pub output_labels: bool,
}

impl<T: BlobType> BaseDataLayerImpl<T> {
    pub fn new(param: &LayerParameter) -> Self {
        BaseDataLayerImpl {
            layer: LayerImpl::new(param),
            transform_param: param.get_transform_param().clone(),
            data_transformer: None,
            output_labels: false,
        }
    }
}

pub trait BaseDataLayer {
    type BaseDataType: BlobType;

    fn get_data_impl(&self) -> &BaseDataLayerImpl<Self::BaseDataType>;

    fn get_data_impl_mut(&mut self) -> &mut BaseDataLayerImpl<Self::BaseDataType>;

    fn data_layer_setup(&mut self, bottom: &BlobVec<Self::BaseDataType>, top: &BlobVec<Self::BaseDataType>);

    /// LayerSetUp: implements common data layer setup functionality, and calls
    /// `data_layer_setUp` to do special data layer setup for individual layer types.
    /// This method may not be overridden except by the `BasePrefetchingDataLayer`.
    fn layer_setup(&mut self, bottom: &BlobVec<Self::BaseDataType>, top: &BlobVec<Self::BaseDataType>) {
        let data = self.get_data_impl_mut();
        data.output_labels = top.len() != 1;
        data.data_transformer = Some(DataTransformer::new(&data.transform_param, data.layer.phase));
        data.data_transformer.as_mut().unwrap().init_rand();

        // The subclasses should setup the size of bottom and top
        self.data_layer_setup(bottom, top);
    }
}


#[derive(Default)]
pub struct Batch<T: BlobType> {
    pub data: ArcBlob<T>,
    pub label: ArcBlob<T>,
}

pub struct BasePrefetchingDataLayerImpl<T: BlobType> {
    pub base: BaseDataLayerImpl<T>,

    pub prefetch: Vec<Batch<T>>,
    pub prefetch_free: BlockingQueue<Batch<T>>,
    pub prefetch_full: BlockingQueue<Batch<T>>,
    pub prefetch_current: Option<Batch<T>>,

    pub transformed_data: Blob<T>,

    pub thread: InternalThreadImpl,
}

impl<T: BlobType> BasePrefetchingDataLayerImpl<T> {
    pub fn new(param: &LayerParameter) -> Self {
        let prefetch_count = param.get_data_param().get_prefetch() as usize;
        let mut this = Self {
            base: BaseDataLayerImpl::new(param),
            prefetch: Vec::with_capacity(prefetch_count),
            prefetch_free: BlockingQueue::new(),
            prefetch_full: BlockingQueue::new(),
            prefetch_current: None,
            transformed_data: Blob::new(),
            thread: InternalThreadImpl::default(),
        };
        this.prefetch.resize_with(prefetch_count, Default::default);

        this
    }
}

pub trait BasePrefetchingDataLayer: BaseDataLayer {
    type PrefetchDataType: Send + 'static;

    fn get_prefetch(&self) -> &BasePrefetchingDataLayerImpl<Self::BaseDataType>;

    fn get_prefetch_mut(&mut self) -> &mut BasePrefetchingDataLayerImpl<Self::BaseDataType>;

    fn forward_cpu(&mut self, _bottom: &BlobVec<Self::BaseDataType>, top: &BlobVec<Self::BaseDataType>) {
        let base = self.get_prefetch_mut();
        base.prefetch_current.take().map(|b| base.prefetch_free.push(b));
        let mut batch = base.prefetch_full.pop();
        // Reshape to loaded data.
        let t0 = &top[0];
        let mut data = std::mem::take(&mut batch.data).into_blob().ok().unwrap();
        t0.borrow_mut().reshape_like(&data);
        t0.borrow_mut().set_cpu_data(&data.cpu_data_shared());
        if base.base.output_labels {
            let mut label = std::mem::take(&mut batch.label).into_blob().ok().unwrap();
            let t1 = &top[1];
            t1.borrow_mut().reshape_like(&label);
            t1.borrow_mut().set_cpu_data(&label.cpu_data_shared());
        }

        // `batch` value are taken and only leaving default value.
        base.prefetch_current.replace(batch);
    }

    fn forward_gpu(&mut self, bottom: &BlobVec<Self::BaseDataType>, top: &BlobVec<Self::BaseDataType>) {
        no_gpu!();
    }

    fn get_sync_data(&self) -> Self::PrefetchDataType;

    fn load_batch(data: &mut Self::PrefetchDataType, batch: &mut Batch<Self::BaseDataType>) where Self: Sized;
}


pub struct ThreadEntryData<T: BlobType, S: Send + 'static> {
    pub prefetch_free: BlockingQueue<Batch<T>>,
    pub prefetch_full: BlockingQueue<Batch<T>>,
    pub prefetch_data: S,
}

impl<U> InternalThread for U
    where
        U: BasePrefetchingDataLayer {
    type EntryData = ThreadEntryData<U::BaseDataType, U::PrefetchDataType>;

    fn get_thread(&self) -> &InternalThreadImpl {
        &self.get_prefetch().thread
    }

    fn get_thread_mut(&mut self) -> &mut InternalThreadImpl {
        &mut self.get_prefetch_mut().thread
    }

    fn get_entry_data(&mut self) -> Box<Self::EntryData> {
        let base = self.get_prefetch();
        let data = self.get_sync_data();
        Box::new(ThreadEntryData {
            prefetch_free: base.prefetch_free.clone(),
            prefetch_full: base.prefetch_full.clone(),
            prefetch_data: data
        })
    }

    fn internal_thread_entry(token: CancelToken, data: Box<Self::EntryData>) {
        let mut prefetch_data = data.prefetch_data;
        let prefetch_free = data.prefetch_free;
        let prefetch_full = data.prefetch_full;
        while !token.is_cancelled() {
            let mut batch = prefetch_free.pop();
            Self::load_batch(&mut prefetch_data, &mut batch);
            prefetch_full.push(batch);
        }
    }
}


pub trait BasePrefetchingDataLayerSetup {
    type BlobDataType: BlobType;

    fn layer_setup(&mut self, bottom: &BlobVec<Self::BlobDataType>, top: &BlobVec<Self::BlobDataType>);
}

impl<T> BasePrefetchingDataLayerSetup for T
    where
        T: BasePrefetchingDataLayer {
    type BlobDataType = T::BaseDataType;

    fn layer_setup(&mut self, bottom: &BlobVec<Self::BlobDataType>, top: &BlobVec<Self::BlobDataType>) {
        BaseDataLayer::layer_setup(self, bottom, top);

        let base = self.get_prefetch_mut();

        // Before starting the prefetch thread, we make cpu_data and gpu_data
        // calls so that the prefetch thread does not accidentally make simultaneous
        // memory alloc calls when the main thread is running.
        let output_labels = base.base.output_labels;
        for prefetch in &mut base.prefetch {
            let mut batch = std::mem::take(prefetch);

            let mut data = batch.data.into_blob().ok().unwrap();
            data.mutable_cpu_data();
            batch.data = ArcBlob::from(data).ok().unwrap();
            if output_labels {
                let mut labels = batch.label.into_blob().ok().unwrap();
                labels.mutable_cpu_data();
                batch.label = ArcBlob::from(labels).ok().unwrap();
            }

            *prefetch = batch;
        }

        info!("Initializing prefetch.");
        // base.base.data_transformer.unwrap().init_rand();
        self.start_internal_thread();
        info!("Prefetch initialized.")
    }
}
