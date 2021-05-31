use crate::blob::{BlobType, Blob, ArcBlob};
use crate::layer::{LayerImpl, BlobVec};
use crate::proto::caffe::{TransformationParameter, LayerParameter};
use crate::data_transformer::DataTransformer;
use crate::util::blocking_queue::BlockingQueue;
use crate::internal_thread::{InternalThread, CancelToken, InternalThreadImpl};


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
    type DataType: BlobType;

    fn get_data_impl(&self) -> &BaseDataLayerImpl<Self::DataType>;

    fn get_data_impl_mut(&mut self) -> &mut BaseDataLayerImpl<Self::DataType>;

    fn data_layer_setup(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>);

    /// LayerSetUp: implements common data layer setup functionality, and calls
    /// `data_layer_setUp` to do special data layer setup for individual layer types.
    /// This method may not be overridden except by the `BasePrefetchingDataLayer`.
    fn layer_setup(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
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
        };
        this.prefetch.resize_with(prefetch_count, Default::default);

        this
    }
}

pub trait BasePrefetchingDataLayer: BaseDataLayer + InternalThread {
    type SyncType: Sync + 'static;

    fn get_prefetch(&self) -> &BasePrefetchingDataLayerImpl<Self::DataType>;

    fn get_prefetch_mut(&mut self) -> &mut BasePrefetchingDataLayerImpl<Self::DataType>;

    fn layer_setup(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
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

            std::mem::replace(prefetch, batch);
        }

        info!("Initializing prefetch.");
        // base.base.data_transformer.unwrap().init_rand();
        self.start_internal_thread();
        info!("Prefetch initialized.")
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        //
    }

    fn forward_gpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        //
    }

    fn get_sync_data(&self) -> &Self::SyncType;

    fn load_batch(data: &Self::SyncType, batch: &mut Batch<Self::DataType>) where Self: Sized;
}


pub struct ThreadEntryData<T: BlobType> {
    pub prefetch_free: BlockingQueue<Batch<T>>,
    pub prefetch_full: BlockingQueue<Batch<T>>,
}

impl<U> InternalThread for U
    where
        U: BasePrefetchingDataLayer {
    type EntryData = ThreadEntryData<U::DataType>;

    fn get_thread(&self) -> &InternalThreadImpl {
        todo!()
    }

    fn get_thread_mut(&mut self) -> &mut InternalThreadImpl {
        todo!()
    }

    fn get_entry_data(&mut self) -> Box<Self::EntryData> {
        todo!()
    }

    fn internal_thread_entry(token: CancelToken, data: Box<Self::EntryData>) {
        todo!()
    }
}


